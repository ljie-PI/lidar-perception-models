#!/usr/bin/env python

import os
import sys
import math
import argparse
import logging
import torch
import torchvision
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from google.protobuf import text_format
from dataset import PointPillarsDataset, create_data_loader, create_eval_data_loader
from optimizer import create_optimizer
from pointpillars import PointPillarsScatter, create_model, draw_model_graph
from model_util import latest_checkpoint, save_model, restore_model
from box_ops import decode_box_torch, center_to_minmax_2d_torch, BOX_ENCODE_SIZE
from generated import pp_config_pb2


def backup_config(config_file, model_path):
    conf_base_file = os.path.basename(config_file)
    target_file = os.path.join(model_path, conf_base_file)
    if os.path.exists(target_file):
        return
    os.system("cp {} {}".format(config_file, target_file))


def update_summary(sum_writer, metrics, global_step):
    for key, val in metrics.items():
        if isinstance(val, (list, tuple)):
            val = {str(i): e for i, e in enumerate(val)}
            if sum_writer is not None:
                sum_writer.add_scalars(key, val, global_step)
        else:
            if sum_writer is not None:
                sum_writer.add_scalar(key, val, global_step)


def log_metrics(metrics):
    metrics_str_list = []
    for key, val in metrics.items():
        if isinstance(val, float):
            metrics_str_list.append("{}={:.3}".format(key, val))
        elif isinstance(val, (list, tuple)):
            if val and isinstance(val[0], float):
                val_str = ', '.join(["{:.3}".format(e) for e in val])
                metrics_str_list.append("{}=[{}]".format(key, val_str))
            else:
                metrics_str_list.append("{}={}".format(key, val))
        else:
            metrics_str_list.append("{}={}".format(key, val))
    logging.info(', '.join(metrics_str_list))


def example_convert_to_torch(example, dtype=torch.float32, device=None):
    device = device or torch.device("cuda")
    example_torch = {}
    float_names = {"voxel_data", "reg_targets"}
    int_names = {"voxel_coord", "cls_targets", "dir_targets"}
    long_names = {"anchor_indices", "example_id"}
    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in int_names:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device=device)
        elif k in long_names:
            example_torch[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        else:
            example_torch[k] = v
    return example_torch


def train_one_step(train_config, model, optimizer, example_torch,
                   model_grid_size, sum_writer, global_epoch, global_step):
    batch_size = example_torch["cls_targets"].shape[0]

    voxel_data = example_torch["voxel_data"]
    voxel_coord = example_torch["voxel_coord"]
    anchor_indices = example_torch["anchor_indices"]
    cls_targets = example_torch["cls_targets"]
    reg_targets = example_torch["reg_targets"]
    dir_targets = example_torch["dir_targets"]
    # logging.info("(PointPillars/training): shape of voxel_data is: {}".format(voxel_data.shape))
    # logging.info("(PointPillars/training): shape of voxel_coord is: {}".format(voxel_coord.shape))
    # logging.info("(PointPillars/training): shape of anchor_indices is: {}".format(anchor_indices.shape))
    # logging.info("(PointPillars/training): shape of cls_targets is: {}".format(cls_targets.shape))
    # logging.info("(PointPillars/training): shape of reg_targets is: {}".format(reg_targets.shape))
    # logging.info("(PointPillars/training): shape of dir_targets is: {}".format(dir_targets.shape))

    # forward
    (loss, cls_loss, box_loss, dir_loss, num_pos, cls_preds_map, cls_preds, box_preds, dir_preds) =\
        model(voxel_data, voxel_coord, anchor_indices, cls_targets, reg_targets, dir_targets,
              return_preds=train_config.enable_summary)

    # backward
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    # update metrics
    metrics = {
        "epoch": global_epoch,
        "steps": global_step,
        "loss": loss.sum(),
        "cls_loss": cls_loss.sum(),
        "box_loss": box_loss.sum(),
        "dir_loss": dir_loss.sum(),
        "num_pos": num_pos.sum(),
        "lr": float(optimizer.param_groups[0]['lr'])
    }
    if global_step > 0 and global_step % train_config.steps_to_update_metric == 0:
        update_summary(sum_writer, metrics, global_step)
        log_metrics(metrics)
        for name, param in model.named_parameters():
            sum_writer.add_histogram(name, param, global_step=global_step)
            if param.grad is not None:
                sum_writer.add_histogram(name + "_grad", param.grad, global_step=global_step)
        if train_config.enable_summary:
            # predict and label count for each class
            cls_max_score, cls_max_idx = torch.max(cls_preds, dim=-1)
            cls_0_pred_cnt = (cls_max_score < 0.5).sum()
            cls_1_pred_cnt = ((cls_max_score >= 0.5) & (cls_max_idx == 0)).sum()
            cls_2_pred_cnt = ((cls_max_score >= 0.5) & (cls_max_idx == 1)).sum()
            cls_3_pred_cnt = ((cls_max_score >= 0.5) & (cls_max_idx == 2)).sum()
            cls_4_pred_cnt = ((cls_max_score >= 0.5) & (cls_max_idx == 3)).sum()
            cls_5_pred_cnt = ((cls_max_score >= 0.5) & (cls_max_idx == 4)).sum()
            sum_writer.add_scalars(
                "class_pred_count",
                {"class_0": cls_0_pred_cnt,
                 "class_1": cls_1_pred_cnt,
                 "class_2": cls_2_pred_cnt,
                 "class_3": cls_3_pred_cnt,
                 "class_4": cls_4_pred_cnt,
                 "class_5": cls_5_pred_cnt},
                global_step=global_step
            )

            cls_0_label_cnt = (cls_targets == 0).sum()
            cls_1_label_cnt = (cls_targets == 1).sum()
            cls_2_label_cnt = (cls_targets == 2).sum()
            cls_3_label_cnt = (cls_targets == 3).sum()
            cls_4_label_cnt = (cls_targets == 4).sum()
            cls_5_label_cnt = (cls_targets == 5).sum()
            sum_writer.add_scalars(
                "class_label_count",
                {"class_0": cls_0_label_cnt,
                 "class_1": cls_1_label_cnt,
                 "class_2": cls_2_label_cnt,
                 "class_3": cls_3_label_cnt,
                 "class_4": cls_4_label_cnt,
                 "class_5": cls_5_label_cnt},
                global_step=global_step
            )
            # add input and class prediction for debug
            voxel_debug_scatter = PointPillarsScatter(
                model_grid_size, num_input_features=1)
            voxel_debug = torch.ones([batch_size, voxel_data.shape[1], 1])
            voxel_img = voxel_debug_scatter(voxel_debug, voxel_coord, batch_size)
            sum_writer.add_images("non_empty_voxels", voxel_img, global_step=global_step)
            max_pred_scores = torch.max(cls_preds_map, dim=-1)[0]
            fg_pred_img = (max_pred_scores >= 0.5).type(torch.float32).unsqueeze(1)
            sum_writer.add_images("fg_pred_image", fg_pred_img, global_step=global_step)


def parse_anchors(anchor_indices, anchor_sizes, len_anchor_size, anchor_rots, len_anchor_rot,
                  x_size, y_size, z_size, resolution, min_offset):
    """ parse anchors according to anchor_indices and anchor_config
    index of anchor is calculated by:
        ((((y * x_size + x) * z_size) + z) * len(anchor_size) + anchor_size_offset) * len(anchor_rot) + anchor_rot
    """
    anchor_rot_idx = anchor_indices % len_anchor_rot
    anchor_rot = anchor_rots[anchor_rot_idx].unsqueeze(-1)

    anchor_indices /= len_anchor_rot
    anchor_size_idx = anchor_indices % len_anchor_size
    anchor_size = anchor_sizes[anchor_size_idx]

    anchor_indices /= len_anchor_size
    z_idx = anchor_indices % z_size
    anchor_indices /= z_size
    y_idx = anchor_indices % y_size
    anchor_indices /= y_size
    x_idx = anchor_indices % x_size
    anchor_pos = torch.cat([x_idx.unsqueeze(-1), y_idx.unsqueeze(-1), z_idx.unsqueeze(-1)],
                            axis=-1).type(torch.float32)
    anchor_pos = (anchor_pos + 0.5) * resolution.unsqueeze(0) + min_offset

    anchors = torch.cat([anchor_pos, anchor_size, anchor_rot], axis=-1)
    return anchors


def predict(model, data_loader, pred_output, config):
    model.eval()

    model_device = next(model.parameters()).device
    anchor_sizes = [[a.length, a.width, a.height] for a in list(config.anchor_config.anchor_size)]
    anchor_sizes = torch.tensor(anchor_sizes, device=model_device).type(torch.float32)
    len_anchor_size = anchor_sizes.shape[0]
    anchor_rots = torch.tensor([0.0, math.pi / 2], device=model_device).type(torch.float32)
    len_anchor_rot = anchor_rots.shape[0]
    vxconf = config.voxel_config
    x_size = math.ceil((vxconf.x_range_max - vxconf.x_range_min) / vxconf.x_resolution)
    y_size = math.ceil((vxconf.y_range_max - vxconf.y_range_min) / vxconf.y_resolution)
    z_size = math.ceil((vxconf.z_range_max - vxconf.z_range_min) / vxconf.z_resolution)
    resolution = torch.tensor([vxconf.x_resolution,
                               vxconf.y_resolution,
                               vxconf.z_resolution],
                              device=model_device).type(torch.float32)
    min_offset = torch.tensor([vxconf.x_range_min,
                               vxconf.y_range_min,
                               vxconf.z_range_min],
                              device=model_device).type(torch.float32)

    for example in iter(data_loader):
        example = example_convert_to_torch(example)
        batch_size = example["voxel_data"].shape[0]
        voxel_data = example["voxel_data"]
        voxel_coord = example["voxel_coord"]
        batch_example_ids = example["example_id"].view(batch_size, 1)
        (batch_cls_preds, batch_box_preds, batch_dir_preds) = model(voxel_data, voxel_coord)
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, config.model_config.num_class)
        batch_box_preds = batch_box_preds.view(batch_size, -1, BOX_ENCODE_SIZE)

        if config.model_config.use_dir_class:
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        for box_preds, cls_preds, dir_preds, example_id in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_example_ids):
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            total_scores = torch.sigmoid(cls_preds)

            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            top_scores, top_labels = torch.max(total_scores, dim=-1)
            # filter boxes with score larger than nms_score_threshold and execute nms
            nms_score_threshold = config.model_config.nms_score_threshold
            if nms_score_threshold > 0.0:
                thresh = torch.tensor([nms_score_threshold],
                                      device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if nms_score_threshold > 0.0:
                    top_labels = top_labels[top_scores_keep]
                    box_preds = box_preds[top_scores_keep]
                    anchor_indices = torch.where(top_scores_keep)[0]
                    anchors = parse_anchors(
                        anchor_indices,
                        anchor_sizes,
                        len_anchor_size,
                        anchor_rots,
                        len_anchor_rot,
                        x_size,
                        y_size,
                        z_size,
                        resolution,
                        min_offset
                    )
                    box_preds = decode_box_torch(box_preds, anchors)
                    if config.model_config.use_dir_class:
                        dir_labels = dir_labels[top_scores_keep]

                
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                # box_preds_corners = center_to_corner_box2d(
                #     boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                # boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                boxes_for_nms = center_to_minmax_2d_torch(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4])

                # logging.info("(model_run/predict): shape of boxes_for_nms: {}".format(boxes_for_nms.shape))
                # logging.info("(model_run/predict): shape of top_scores: {}".format(top_scores.shape))
                # the nms in 3d detection just remove overlap boxes.
                selected = torchvision.ops.nms(boxes_for_nms, top_scores, config.model_config.nms_iou_threshold)
            else:
                selected = None

            # finally generate predictions.
            if selected is not None:
                selected_boxes = box_preds[selected]
                if config.model_config.use_dir_class:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if config.model_config.use_dir_class:
                    opp_labels = (box_preds[..., -1] > 0) ^ (selected_dir_labels > 0)
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                label_len = label_preds.shape[0]
                assert label_len == scores.shape[0]
                assert label_len == box_preds.shape[0]
                res_file = str(example_id.item()) + ".pred"
                with open(os.path.join(pred_output, res_file), "w") as fpred:
                    for i in range(label_len):
                        fpred.write("{} {} {} {} {} {} {} {} {}\n".format(
                            box_preds[i, 0], box_preds[i, 1], box_preds[i, 2], box_preds[i, 3],
                            box_preds[i, 4], box_preds[i, 5], box_preds[i, 6], label_preds[i], scores[i]))


def model_train(config_file, train_data_path, eval_data_path, model_path):
    logging.info("\n\n\nBegin to train model:")
    logging.info("***** config_file = {}".format(config_file))
    logging.info("***** train_data_path = {}".format(train_data_path))
    logging.info("***** eval_data_path = {}".format(eval_data_path))
    logging.info("***** model_path = {}".format(model_path))

    # create model_path and backup config file
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    backup_config(config_file, model_path)

    config = pp_config_pb2.PointPillarsConfig()
    fconfig = open(config_file)
    text_format.Parse(fconfig.read(), config)
    fconfig.close()
    train_config = config.train_config
    model_config = config.model_config

    data_set = PointPillarsDataset(config, train_data_path, is_train=True)
    data_loader = create_data_loader(config, data_set)
    eval_data_set = PointPillarsDataset(config, eval_data_path, is_train=False)
    eval_data_loader = create_eval_data_loader(config, eval_data_set)

    sum_writer = None
    if train_config.enable_summary:
        summary_dir = os.path.join(model_path, "summary")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        sum_writer = SummaryWriter(log_dir=str(summary_dir))

    model = create_model(config)
    model.cuda()
    if sum_writer is not None:
        data_iter = iter(data_loader)
        example = next(data_iter)
        example_torch = example_convert_to_torch(example)
        draw_model_graph(model, sum_writer, example_torch)
    model_grid_size = model.dense_shape
    if train_config.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = create_optimizer(model, train_config)

    ckpt_file, last_step, last_epoch = latest_checkpoint(model_path, model_config.model_name)
    logging.info("Latest checkpoint file: {}".format(ckpt_file))
    if ckpt_file is not None:
        logging.info("Will restore model from checkpoint file: {}".format(ckpt_file))
        restore_model(model, ckpt_file)
        logging.info("Model restored")

    total_train_steps = train_config.train_epochs * len(data_set) // train_config.batch_size
    train_eval_loops = 1
    if train_config.steps_per_eval > 0:
        train_eval_loops = total_train_steps // train_config.steps_per_eval
        if total_train_steps % train_config.steps_per_eval != 0:
            train_eval_loops += 1

    global_step = 0
    global_step += last_step
    global_epoch = last_epoch
    logging.info("***** batch_size = {:d}".format(train_config.batch_size))
    logging.info("***** epoch = {:d}".format(train_config.train_epochs))
    logging.info("***** total_train_steps = {:d}".format(total_train_steps))
    logging.info("***** train_eval_loops = {:d}".format(train_eval_loops))
    logging.info("***** global_step = {:d}".format(global_step))
    logging.info("***** last_epoch = {:d}".format(last_epoch))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, train_config.lr_decay, last_epoch=-1)

    try:
        data_iter = iter(data_loader)
        for _ in range(train_eval_loops):
            model.train()
            if global_step + train_config.steps_per_eval > total_train_steps:
                steps = total_train_steps % train_config.steps_per_eval
            else:
                steps = train_config.steps_per_eval
            for step in range(steps):
                global_step += 1
                try:
                    example = next(data_iter)
                except StopIteration:
                    global_epoch += 1
                    lr_scheduler.step()
                    data_iter = iter(data_loader)
                    example = next(data_iter)
                example_torch = example_convert_to_torch(example)
                train_one_step(train_config, model, optimizer, example_torch,
                               model_grid_size, sum_writer, global_epoch, global_step)
                if global_step > 0 and global_step % train_config.steps_to_save_ckpts == 0:
                    save_model(
                        model_dir=model_path,
                        model=model,
                        model_name=model_config.model_name,
                        global_step=global_step,
                        global_epoch=global_epoch,
                        max_to_keep=train_config.max_keep_ckpts)

            logging.info("\n############### predicting({:d} steps) ###############".format(global_step))
            pred_output = os.path.join(model_path, "eval-res-{:d}".format(global_step))
            if not os.path.exists(pred_output):
                os.makedirs(pred_output)
            predict(model, eval_data_loader, pred_output, config)
        
        # save model after train
        save_model(
            model_dir=model_path,
            model=model,
            model_name=model_config.model_name,
            global_step=global_step,
            global_epoch=global_epoch,
            max_to_keep=train_config.max_keep_ckpts)

    except Exception as e:
        save_model(
            model_dir=model_path,
            model=model,
            model_name=model_config.model_name,
            global_step=global_step,
            global_epoch=global_epoch,
            max_to_keep=train_config.max_keep_ckpts)
        raise e


def model_predict(config_file, pred_data_path, pred_output, model_path):
    logging.info("Begin to predict:")
    logging.info("config_file = {}".format(config_file))
    logging.info("pred_data_path = {}".format(pred_data_path))
    logging.info("pred_output = {}".format(pred_output))
    logging.info("model_path = {}".format(model_path))

    if not os.path.exists(model_path):
        raise Exception("model_path: {} doesn't exist".format(model_path))

    config = pp_config_pb2.PointPillarsConfig()
    fconfig = open(config_file)
    text_format.Parse(fconfig.read(), config)
    fconfig.close()
    train_config = config.train_config
    model_config = config.model_config

    model = create_model(config)
    model_grid_size = model.dense_shape
    if train_config.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    ckpt_file, _, _ = latest_checkpoint(model_path, model_config.model_name)
    logging.info("Latest checkpoint file: {}".format(ckpt_file))
    if ckpt_file is None:
        raise Exception("Failed to get latest checkpoint file in {}".format(model_path))
    logging.info("Will restore model from checkpoint file: {}".format(ckpt_file))
    restore_model(model, ckpt_file)
    logging.info("Model restored")

    data_set = PointPillarsDataset(config, pred_data_path, is_train=False)
    data_loader = create_data_loader(config, data_set)

    predict(model, data_loader, pred_output, config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser();
    argparser.add_argument("--action", default="train", required=True)
    argparser.add_argument("--config", default="./config/pp_config.proto", required=True)
    argparser.add_argument("--train_data_path", default="./data/train_data")
    argparser.add_argument("--eval_data_path", default="./data/eval_data")
    argparser.add_argument("--pred_data_path", default="./data/pred_data")
    argparser.add_argument("--pred_output", default="./data/pred_output")
    argparser.add_argument("--model_path", default="./data/models/model_xxx", required=True)
    argparser.add_argument("--log_level", default="info")
    args = argparser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    log_level = logging.INFO
    if args.log_level.lower() == "debug":
        log_level = logging.DEBUG
    elif args.log_level.lower() == "warning":
        log_level = logging.WARNING
    elif args.log_level.lower() == "error":
        log_level = logging.ERROR
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s',
                        filename=os.path.join(args.model_path, "model.log"),
                        level=log_level)

    if args.action == "train":
        model_train(args.config, args.train_data_path, args.eval_data_path, args.model_path)
    elif args.action == "predict" or args.action == "pred":
        model_predict(args.config, args.pred_data_path, args.pred_output, args.model_path)
    else:
        logging.error("action should be one of `train' or `predict'")
        sys.exit(-1)
