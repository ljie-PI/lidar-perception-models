#!/usr/bin/env python

import os
import sys
import math
import argparse
import logging
import torch
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from google.protobuf import text_format
from dataset import PointPillarsDataset, create_data_loader
from pointpillars import create_model, create_optimizer
from model_util import latest_checkpoint, save_model, restore_model
from box_ops import decode_box_torch, center_to_minmax_2d
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
            sum_writer.add_scalars(key, val, global_step)
        else:
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
        metrics_str_list.append("{} = {}".format(key, val_str))
        logging.info(', '.join(metrics_str_list))


def example_convert_to_torch(example, dtype=torch.float32, device=None):
    device = device or torch.device("cuda")
    example_torch = {}
    float_names = {"voxel_data", "label_data", "anchor_data"}
    int_names = {"voxel_coord", "label_type", "anchor_targets"}
    bool_names = {"anchor_positive"}
    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in int_names:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device=device)
        elif k in bool_names:
            example_torch[k] = torch.as_tensor(v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


def train_one_step(train_config, model, optimizer, example, sum_writer, global_step):
    example_torch = example_convert_to_torch(example)
    batch_size = example["anchors"].shape[0]

    # forward
    ret_dict = model(example_torch)
    loss = ret_dict["loss"]

    # backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    optimizer.zero_grad()
    model.update_global_step()

    # update metrics
    metrics = {
        "loss": float(loss[0]),
        "cls_loss": float(ret_dict["cls_loss_reduced"][0]),
        "loc_loss": float(ret_dict["loc_loss_reduced"][0]),
        "dir_loss": float(ret_dict["dir_loss_reduced"][0]),
        "num_pos": int(ret_dict["num_pos"][0]),
        "lr": float(optimizer.param_groups[0]['lr'])
    }
    update_summary(sum_writer, metrics, global_step)
    if global_step % train_config.display_step == 0:
        log_metrics(metrics)


def parse_anchors(anchor_indices, config, dtype=torch.float32):
    """ parse anchors according to anchor_indices and anchor_config
    index of anchor is calculated by:
        ((((y * x_size + x) * z_size) + z) * len(anchor_size) + anchor_size_offset) * len(anchor_rot) + anchor_rot
    """
    anchor_sizes = torch.tensor(list(anchor_config.anchor_size),
                                device=anchor_indices.device).type(dtype)
    len_anchor_size = anchor_sizes.shape[0]
    anchor_rots = torch.tensor([0, math.pi / 2],
                               device=anchor_indices.device).type(dtype)
    len_anchor_rot = anchor_rots.shape[0]
    vxconf = config.voxel_config
    x_size = math.ceil((vxconf.x_range_max - vxconf.x_range_min) / vxconf.x_resolution)
    y_size = math.ceil((vxconf.y_range_max - vxconf.y_range_min) / vxconf.y_resolution)
    z_size = math.ceil((vxconf.z_range_max - vxconf.z_range_min) / vxconf.z_resolution)
    resolution = torch.tensor([vxconf.x_resolution,
                               vxconf.y_resolution,
                               vxconf.z_resolution],
                              device=anchor_indices.device).type(dtype)
    
    anchor_rot_idx = anchor_indices % len_anchor_rot
    anchor_rot = anchor_rots[anchor_rot_idx].unsqueeze(-1)

    anchor_indices /= len_anchor_rot
    anchor_size_idx = anchor_indices % len_anchor_size
    anchor_size = anchor_size[anchor_size_idx].unsqueeze(-1)

    anchor_indices /= len_anchor_size
    z_idx = anchor_indices % z_size
    anchor_indices /= z_size
    x_idx = anchor_indices % x_size
    anchor_indices /= x_size
    y_idx = anchor_indices % y_size
    anchor_pos = torch.cat([x_idx.unsqueeze(-1), y_idx.unsqueeze(-1), z_idx.unsqueeze(-1)],
                            axis=-1).type(dtype)
    anchor_pos = (anchor_pos + 0.5) * resolution.unsqueeze(0)
    anchors = torch.cat([anchor_pos, anchor_size, anchor_rot], axis=-1)
    return anchors


def predict(model, data_loader, pred_output, config):
    model.eval()
    for example in iter(data_loader):
        example = example_convert_to_torch(example)
        batch_size = example['voxel_data'].shape[0]
        preds_dict = model(example)
        batch_cls_preds = preds_dict["cls_preds"]
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, config.model_config.num_class)
        batch_box_preds = preds_dict["box_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1, BOX_ENCODE_SIZE)

        if model.use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds):
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            total_scores = torch.sigmoid(cls_preds)

            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            top_scores, top_labels = torch.max(total_scores, dim=-1)
            # filter boxes with score larger than nms_score_threshold and execute nms
            nms_score_threshold = self._config.nms_score_threshold
            if nms_score_threshold > 0.0:
                thresh = torch.tensor([nms_score_threshold],
                                      device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if nms_score_threshold > 0.0:
                    top_labels = top_labels[top_scores_keep]
                    box_preds = box_preds[top_scores_keep]
                    anchor_indices = torch.where(top_scores_keep)
                    anchors = parse_anchors(anchor_indices, config, dtype=box_preds.type())
                    box_preds = decode_box_torch(box_preds, anchors)
                    if model.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                # box_preds_corners = center_to_minmax_2d(
                #     boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                # boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                boxes_for_nms = center_to_minmax_2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 3:5])

                # the nms in 3d detection just remove overlap boxes.
                selected = torchvision.ops.nms(boxes_for_nms, top_scores, config.model_config.nms_iou_threshold)
            else:
                selected = None

            # finally generate predictions.
            if selected is not None:
                selected_boxes = box_preds[selected]
                if model.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if model.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))

                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                # predictions
                predictions_dict = {
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds
                }
            else:
                predictions_dict = {
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


def model_train(config_file, train_data_path, eval_data_path, model_path):
    logging.info("Begin to train model:")
    logging.info("config_file = {}".format(config_file))
    logging.info("train_data_path = {}".format(train_data_path))
    logging.info("eval_data_path = {}".format(eval_data_path))
    logging.info("model_path = {}".format(model_path))

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

    model = create_model(config)
    model.cuda()
    optimizer = create_optimizer(model, train_config)
    lr_scheduler = ExponentialLR(optimizer, train_config.lr_decay, last_epoch=-1)

    ckpt_file = latest_checkpoint(model_path, model_config.model_name)
    if ckpt_file is not None:
        restore_model(model, ckpt_file)

    data_set = PointPillarsDataset(config, train_data_path, is_train=True)
    data_loader = create_data_loader(config, data_set)
    data_iter = iter(data_loader)
    eval_data_set = PointPillarsDataset(config, eval_data_path, is_train=False)
    eval_data_loader = create_data_loader(config, eval_data_set)

    summary_dir = os.path.join(model_path, "summary")
    if not os.path.exists and train_config.enable_summary:
        os.makedirs(summary_dir)
        sum_writer = SummaryWriter(str(summary_dir))

    total_train_steps = train_config.train_epochs * len(data_set) // train_config.batch_size + 1
    train_eval_loops = 1
    if train_config.steps_per_eval > 0:
        train_eval_loops = total_train_steps // train_config.steps_per_eval
        if total_train_steps % train_config.steps_per_eval != 0:
            train_eval_loops += 1

    global_step = 0
    global_epoch = 0
    optimizer.zero_grad()
    try:
        for _ in range(train_eval_loops):
            model.train()
            if global_step + train_config.steps_per_eval > total_train_steps:
                steps = total_train_steps % train_config.steps_per_eval
            else:
                steps = train_config.steps_per_eval
            for step in range(steps):
                lr_scheduler.step()
                try:
                    example = next(data_iter)
                except StopIteration:
                    logging.info("epoch {} finished.".format(global_epoch))
                    global_epoch += 1
                    data_iter = iter(data_loader)
                    example = next(data_iter)
                    global_step += 1
                train_one_step(train_config, model, optimizer, example, sum_writer)
                if global_step % train_config.steps_to_save_chpts == 0:
                    save_model(
                        model_dir=model_path,
                        model=model,
                        model_name=model_config.model_name,
                        global_step=global_step,
                        max_to_keep=model_config.max_keep_chpts)

            logging.info("\n############### predicting ###############")
            pred_output = os.path.join(model_path, "eval-res-{:d}".format(global_step))
            predict(model, eval_data_loader, pred_output, config)
            save_model(
                model_dir=model_path,
                model=model + "_eval",
                model_name=model_config.model_name,
                global_step=global_step,
                max_to_keep=model_config.max_keep_chpts)

    except Exception as e:
        save_model(
            model_dir=model_path,
            model=model,
            model_name=model_config.model_name,
            global_step=global_step,
            max_to_keep=model_config.max_keep_chpts)
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
    model.cuda()

    ckpt_file = latest_checkpoint(model_path, model_config.model_name)
    if ckpt_file is not None:
        raise Exception("Failed to get latest checkpoint file in {}".format(model_path))
        restore_model(model, ckpt_file)

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
    args = argparser.parse_args()

    logging.basicConfig(filename=os.path.join(args.model_path, "model.log"),
                        level=logging.INFO)

    if args.action == "train":
        model_train(args.config, args.train_data_path, args.eval_data_path, args.model_path)
    elif args.action == "predict" or args.action == "pred":
        model_predict(args.config, args.pred_data_path, args.pred_output, args.model_path)
    else:
        logging.error("action should be one of `train' or `predict'")
        sys.exit(-1)