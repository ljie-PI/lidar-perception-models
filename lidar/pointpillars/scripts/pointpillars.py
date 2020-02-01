#!/usr/bin/env python

import math
import logging
import torch

from loss import alloc_loss_weights, create_loss
from box_ops import BOX_ENCODE_SIZE


class PFNLayer(torch.nn.Module):
    """
    Layer to extract features on pillars
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.linear = torch.nn.Linear(in_channels, self.units, bias=False)
        self.norm = torch.nn.BatchNorm1d(self.units)

    def forward(self, inputs):
        # logging.info("(FPNLayer): shape of inputs: {}".format(inputs.shape))
        x = self.linear(inputs)
        # logging.info("(FPNLayer): shape of linear: {}".format(x.shape))
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = torch.nn.functional.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max

        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        return x_concatenated


class PointPillarsFeatureNet(torch.nn.Module):
    """
    Network to extract features on pillars
    """
    def __init__(self, config):
        super(PointPillarsFeatureNet, self).__init__()
        self.name = 'PointPillarsFeatureNet'

        self._num_voxels = config.voxel_config.num_voxels
        self._num_points_per_voxel = config.voxel_config.num_points_per_voxel
        self.input_feat_dim = 3  # [x, y, z] of each points
        if config.model_config.use_reflection:
            self.input_feat_dim += 1
        self.input_feat_dim += 4 # add distance and [x, y, z] to pillar center
        pfn_out_dims = config.model_config.pillar_feat_filters
        num_filters = [self.input_feat_dim] + list(pfn_out_dims)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
                self.out_filters = out_filters
            pfn_layers.append(PFNLayer(in_filters, out_filters, last_layer=last_layer))
        self.pfn_layers = torch.nn.ModuleList(pfn_layers)

    def forward(self, voxel_data):
        # logging.info("(PointPillarsFeatureNet): shape of voxel_data: {}".format(voxel_data.shape))

        features = voxel_data.view([-1, self._num_points_per_voxel, self.input_feat_dim])
        # logging.info("(PointPillarsFeatureNet): shape of features: {}".format(features.shape))

        for pfn in self.pfn_layers:
            features = pfn(features)
        # logging.info("(PointPillarsFeatureNet): shape of features: {}".format(features.shape))

        pillar_feats = features.view([-1, self._num_voxels, self.out_filters])
        # logging.info("(PointPillarsFeatureNet): shape of pillar_feats: {}".format(pillar_feats.shape))
        return pillar_feats


class PointPillarsScatter(torch.nn.Module):
    def __init__(self, dense_shape, num_input_features=64):
        super(PointPillarsScatter, self).__init__()
        self.name = 'PointPillarsScatter'

        self.x_size = dense_shape[0]
        self.y_size = dense_shape[1]
        self.nchannels = num_input_features

    def forward(self, input_feat, coords, batch_size):
        # logging.info("(PointPillarsScatter): shape of input_feat: {}".format(input_feat.shape))
        # logging.info("(PointPillarsScatter): shape of coords: {}".format(coords.shape))
        # logging.info("(PointPillarsScatter): shape of batch_size {:d}".format(batch_size))

        batch_canvas = []
        for batch_idx in range(batch_size):
            canvas = torch.zeros(self.x_size * self.y_size, self.nchannels,
                                 dtype=input_feat.dtype, device=input_feat.device)
            indices = (coords[batch_idx, :, 0] * self.y_size + coords[batch_idx, :, 1]).type(torch.long)
            canvas[indices, :] = input_feat[batch_idx]
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0).permute(0, 2, 1).contiguous()
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.x_size, self.y_size)

        return batch_canvas


class RPN(torch.nn.Module):
    """
    RPN network based on pseudo image feature map
    """
    def __init__(self, model_config, num_input_filters):
        super(RPN, self).__init__()
        layer_nums = list(model_config.rpn_layer_num)
        layer_strides = list(model_config.rpn_layer_strides)
        num_filters = list(model_config.rpn_num_filters)
        upsample_strides = list(model_config.rpn_upsample_strides)
        upsample_filters = list(model_config.rpn_upsample_filters)

        # level 1 feature pyramid
        self.fp_level_1 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_input_filters,
                num_filters[0],
                kernel_size=3,
                stride=layer_strides[0],
                bias=False
            ),
            torch.nn.BatchNorm2d(num_filters[0]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            prefix = "l{:d}_".format(i + 1)
            self.fp_level_1.add_module(
                prefix + "conv",
                torch.nn.Conv2d(
                    num_filters[0],
                    num_filters[0],
                    kernel_size=3,
                    padding=1,
                    bias=False
                ))
            self.fp_level_1.add_module(prefix + "bn", torch.nn.BatchNorm2d(num_filters[0]))
            self.fp_level_1.add_module(prefix + "relu", torch.nn.ReLU())
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[0],
                upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
                bias=False
            ),
            torch.nn.BatchNorm2d(upsample_filters[0]),
            torch.nn.ReLU(),
        )
        # level 2 feature pyramid
        self.fp_level_2 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_filters[0],
                num_filters[1],
                kernel_size=3,
                stride=layer_strides[1],
                bias=False
            ),
            torch.nn.BatchNorm2d(num_filters[1]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            prefix = "l{:d}_".format(i + 1)
            self.fp_level_2.add_module(
                prefix + "conv",
                torch.nn.Conv2d(
                    num_filters[1],
                    num_filters[1],
                    kernel_size=3,
                    padding=1,
                    bias=False
                ))
            self.fp_level_2.add_module(prefix + "bn", torch.nn.BatchNorm2d(num_filters[1]))
            self.fp_level_2.add_module(prefix + "relu", torch.nn.ReLU())
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[1],
                upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
                bias=False
            ),
            torch.nn.BatchNorm2d(upsample_filters[1]),
            torch.nn.ReLU(),
        )
        # level 2 feature pyramid
        self.fp_level_3 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_filters[1],
                num_filters[2],
                kernel_size=3,
                stride=layer_strides[2],
                bias=False
            ),
            torch.nn.BatchNorm2d(num_filters[2]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            prefix = "l{:d}_".format(i + 1)
            self.fp_level_3.add_module(
                prefix + "conv",
                torch.nn.Conv2d(
                    num_filters[2],
                    num_filters[2],
                    kernel_size=3,
                    padding=1,
                    bias=False
                ))
            self.fp_level_3.add_module(prefix + "bn", torch.nn.BatchNorm2d(num_filters[2]))
            self.fp_level_3.add_module(prefix + "relu", torch.nn.ReLU())
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[2],
                upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
                bias=False
            ),
            torch.nn.BatchNorm2d(upsample_filters[2]),
            torch.nn.ReLU(),
        )

        num_cls = model_config.num_anchor_per_loc * model_config.num_class
        self.conv_cls = torch.nn.Conv2d(sum(upsample_filters), num_cls, 1)
        self.conv_box = torch.nn.Conv2d(
            sum(upsample_filters),
            model_config.num_anchor_per_loc * BOX_ENCODE_SIZE,
            kernel_size=1
        )
        self.use_direction_classifier = model_config.use_dir_class
        if self.use_direction_classifier:
            self.conv_dir_cls = torch.nn.Conv2d(
                sum(upsample_filters),
                model_config.num_anchor_per_loc * 2,
                kernel_size=1
            )

    def forward(self, x):
        x = self.fp_level_1(x)
        up1 = self.deconv1(x)
        x = self.fp_level_2(x)
        up2 = self.deconv2(x)
        x = self.fp_level_3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        if self.use_direction_classifier:
            dir_preds = self.conv_dir_cls(x)
            dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_preds"] = dir_preds
        return ret_dict


class PointPillars(torch.nn.Module):
    def __init__(self, dense_shape, config):
        super(PointPillars, self).__init__()
        self._config = config
        self.use_direction_classifier = config.model_config.use_dir_class
        self.dense_shape = dense_shape

        self.pp_feature_net = PointPillarsFeatureNet(self._config)
        self.pp_scatter = PointPillarsScatter(
                dense_shape, num_input_features=self.pp_feature_net.out_filters)

        self.rpn = RPN(self._config.model_config, self.pp_scatter.nchannels)

    def forward(self,
                voxel_data,
                voxel_coord,
                anchor_indices=None,
                cls_targets=None,
                reg_targets=None,
                dir_targets=None,
                return_preds=False):
        batch_size = voxel_data.shape[0]
        voxel_features = self.pp_feature_net(voxel_data)
        # logging.info("(PointPillars): type of voxel_features {}" \
        #     .format(voxel_features.shape))
        dense_features = self.pp_scatter(voxel_features, voxel_coord, batch_size)
        # logging.info("(PointPillars): type of dense_features {}" \
        #     .format(dense_features.shape))
        preds_dict = self.rpn(dense_features)
        box_preds_map = preds_dict["box_preds"]
        cls_preds_map = preds_dict["cls_preds"]
        # logging.info("(PointPillars): shape of box_preds_map is: {}" \
        #     .format(box_preds_map.shape))
        # logging.info("(PointPillars): shape of cls_preds_map is: {}" \
        #     .format(cls_preds_map.shape))
        dir_preds = None
        if self.use_direction_classifier:
            dir_preds_map = preds_dict["dir_preds"]
            # logging.info("(PointPillars): shape of dir_preds_map is: {}" \
            #     .format(dir_preds_map.shape))

        if self.training:
            cls_preds = cls_preds_map.view(batch_size, -1, self._config.model_config.num_class)
            cls_preds = torch.stack([cls_preds[i, anchor_indices[i]] for i in range(batch_size)])
            box_preds = box_preds_map.view(batch_size, -1, BOX_ENCODE_SIZE)
            box_preds = torch.stack([box_preds[i, anchor_indices[i]] for i in range(batch_size)])
            # logging.info("(PointPillars/training): shape of box_preds is: {}" \
            #     .format(box_preds.shape))
            # logging.info("(PointPillars/training): shape of box_targets is: {}" \
            #     .format(reg_targets.shape))
            # logging.info("(PointPillars/training): shape of cls_preds is: {}" \
            #     .format(cls_preds.shape))
            # logging.info("(PointPillars/training): shape of cls_targets is: {}" \
            #     .format(cls_targets.shape))

            if self.use_direction_classifier:
                dir_preds = dir_preds_map.view(batch_size, -1, 2)
                dir_preds = torch.stack(
                    [dir_preds[i, anchor_indices[i]] for i in range(batch_size)])
                # logging.info("(PointPillars/training): shape of dir_preds is: {}" \
                #     .format(dir_preds.shape))
                # logging.info("(PointPillars/training): shape of dir_targets is: {}" \
                #     .format(dir_targets.shape))

            cls_weights, reg_weights, dir_weights = alloc_loss_weights(
                cls_targets,
                pos_cls_weight=self._config.train_config.pos_class_weight,
                neg_cls_weight=self._config.train_config.neg_class_weight,
                dtype=voxel_data.dtype)

            reg_loss, cls_loss, dir_loss = create_loss(
                box_preds=box_preds,
                cls_preds=cls_preds,
                dir_preds=dir_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                dir_targets=dir_targets,
                dir_weights=dir_weights,
                config=self._config
            )

            reg_loss_reduced = reg_loss.sum() / batch_size
            reg_loss_reduced *= self._config.train_config.reg_loss_weight
            cls_loss_reduced = cls_loss.sum() / batch_size
            cls_loss_reduced *= self._config.train_config.cls_loss_weight
            loss = reg_loss_reduced + cls_loss_reduced
            if self.use_direction_classifier:
                dir_loss_reduced = dir_loss.sum() / batch_size
                dir_loss_reduced *= self._config.train_config.dir_loss_weight
                loss += dir_loss_reduced

            num_pos = (cls_targets > 0).float().sum() / batch_size
            res = [loss, cls_loss_reduced, reg_loss_reduced, dir_loss_reduced, num_pos]
            if return_preds:
                res.extend([cls_preds_map, cls_preds, box_preds, dir_preds])
            return tuple(res)

        return (cls_preds_map, box_preds_map, dir_preds_map)


def create_model(config):
    voxel_config = config.voxel_config
    x_size = math.ceil((voxel_config.x_range_max - voxel_config.x_range_min) \
                        / voxel_config.x_resolution)
    y_size = math.ceil((voxel_config.y_range_max - voxel_config.y_range_min) \
                        / voxel_config.y_resolution)
    grid_size = [x_size, y_size]

    model = PointPillars(grid_size, config)

    return model


def draw_model_graph(model, sum_writer, example):
    if sum_writer is None:
        return
    if model.training:
        voxel_data = example["voxel_data"]
        voxel_coord = example["voxel_coord"]
        anchor_indices = example["anchor_indices"]
        cls_targets = example["cls_targets"]
        reg_targets = example["reg_targets"]
        dir_targets = example["dir_targets"]
        # logging.info("(draw_model_graph): shape of voxel_data is: {}".format(voxel_data.shape))
        # logging.info("(draw_model_graph): shape of voxel_coord is: {}".format(voxel_coord.shape))
        # logging.info("(draw_model_graph): shape of anchor_indices is: {}".format(anchor_indices.shape))
        # logging.info("(draw_model_graph): shape of cls_targets is: {}".format(cls_targets.shape))
        # logging.info("(draw_model_graph): shape of reg_targets is: {}".format(reg_targets.shape))
        # logging.info("(draw_model_graph): shape of dir_targets is: {}".format(dir_targets.shape))
        sum_writer.add_graph(
            model,
            (voxel_data, voxel_coord, anchor_indices,
             cls_targets, reg_targets, dir_targets)
        )
        sum_writer.flush()
