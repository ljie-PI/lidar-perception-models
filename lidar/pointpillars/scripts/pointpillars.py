#!/usr/bin/env python

import math
import logging
import torch

from loss import alloc_loss_weights, create_loss
from box_ops import encode_box, BOX_ENCODE_SIZE


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.linear = torch.nn.Linear(in_channels, self.units)
        self.norm = torch.nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        logging.info("(FPNLayer): shape of inputs: ".format(input.shape))
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = torch.nn.functional.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PointPillarsFeatureNet(nn.Module):
    def __init__(self, model_config):
        super(PointPillarsFeatureNet, self).__init__()
        self.name = 'PointPillarsFeatureNet'

        input_feat_dim = 3  # [x, y, z] of each points
        if model_config.use_reflection:
            input_feat_dim += 1
        input_feat_dim += 4 # add distance and [x, y, z] to pillar center
        pfn_out_dims = model_config.pillar_feat_filters
        num_filters = [input_feat_dim] + list(pfn_out_dims)
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

    def forward(self, features, voxel_points, voxel_coord):
        logging.info("(PointPillarsFeatureNet): shape of features: {}".format(features.shape))
        logging.info("(PointPillarsFeatureNet): shape of voxel_points: {}".format(voxel_points.shape))
        logging.info("(PointPillarsFeatureNet): shape of voxel_coord: {}".format(voxel_coord.shape))

        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self, dense_shape, num_input_features=64):
        super(PointPillarsScatter, self).__init__()
        self.name = 'PointPillarsScatter'

        self.x_size = dense_shape[0]
        self.y_size = dense_shape[1]
        self.nchannels = num_input_features

    def forward(self, input_feat, coords, batch_size):
        logging.info("(PointPillarsScatter): shape of input_feat: {}".format(input_feat.shape))
        logging.info("(PointPillarsScatter): shape of coords: {}".format(coords.shape))
        logging.info("(PointPillarsScatter): batch_size {:d}".format(batch_size))

        batch_canvas = []
        for batch_idx in range(batch_size):
            canvas = torch.zeros(self.nchannels, self.x_size * self.y_size,
                                 dtype=input_feat.dtype, device=input_feat.device)
            this_coords = coords[batch_idx]
            indices = this_coords[:, 1] * self.x_size + this_coords[:, 0]
            indices = indices.type(torch.int32)
            voxels = input_feat[batch_idx]
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)
        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.x_size, self.y_size)

        return batch_canvas


class RPN(nn.Module):
    def __init__(self, model_config, input_shape):
        super(RPN, self).__init__()

        num_input_filters = input_shape[2]

        layer_nums = list(model_config.rpn_layer_num)
        layer_strides = list(model_config.rpn_layer_strides)
        num_filters = list(model_config.rpn_num_filters)
        upsample_strides = list(model_config.rpn_upsample_strides)
        upsample_filters = list(model_config.rpn_upsample_filters)

        self.block1 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            torch.nn.BatchNorm2d(num_filters[0]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                torch.nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(torch.nn.BatchNorm2d(num_filters[0]))
            self.block1.add(torch.nn.ReLU())
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[0],
                upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            torch.nn.BatchNorm2d(upsample_filters[0]),
            torch.nn.ReLU(),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_filters[0], num_filters[1], 3, stride=layer_strides[1]),
            torch.nn.BatchNorm2d(num_filters[1]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                torch.nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(torch.nn.BatchNorm2d(num_filters[1]))
            self.block2.add(torch.nn.ReLU())
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[1],
                upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            torch.nn.BatchNorm2d(upsample_filters[1]),
            torch.nn.ReLU(),
        )

        self.block3 = torch.nn.Sequential(
            torch.nn.ZeroPad2d(1),
            torch.nn.Conv2d(
                num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            torch.nn.BatchNorm2d(num_filters[2]),
            torch.nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                torch.nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(torch.nn.BatchNorm2d(num_filters[2]))
            self.block3.add(torch.nn.ReLU())
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                num_filters[2],
                upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            torch.nn.BatchNorm2d(upsample_filters[2]),
            torch.nn.ReLU(),
        )

        num_cls = model_config.num_anchor_per_loc * model_config.num_class
        self.conv_cls = torch.nn.Conv2d(sum(upsample_filters), num_cls, 1)
        self.conv_box = torch.nn.Conv2d(
            sum(upsample_filters),
            model_config.num_anchor_per_loc * BOX_ENCODE_SIZE,
            1)
        self.use_direction_classifier = model_config.use_dir_class
        if self.use_direction_classifier:
            self.conv_dir_cls = torch.nn.Conv2d(
                sum(upsample_filters), model_config.num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
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
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict

    
class PointPillars(nn.Module):
    def __init__(self, dense_shape, config):
        super(PointPillars, self).__init__()
        self._config = config.model_config
        self._voxel_config = config.voxel_config

        self.pp_feature_net = PointPillarsFeatureNet(self._config)
        self.pp_scatter = PointPillarsScatter(output_shape=dense_shape,
                                              num_input_features=pp_feature_net.out_filters)

        rpn_input_shape = [dense_shape] + list(self.middle_feature_extractor.nchannels)
        self.rpn = RPN(self._config, rpn_input_shape)

    def forward(self, example):
        voxel_data = example["voxel_data"]
        voxel_coord = example["voxel_coord"]
        voxel_points = example["voxel_points"]
        batch_size = voxel_data.shape[0]
        voxel_features = self.pp_feature_net(voxel_data, voxel_points, voxel_coord)
        dense_features = self.pp_scatter(voxel_features, voxel_coord, batch_size)
        preds_dict = self.rpn(dense_features)
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]

        if self.training:
            anchor_indices = example["anchor_indices"]
            cls_targets = example["cls_targets"]
            reg_targets = example["reg_targets"]

            cls_preds = cls_preds.view(batch_size, -1, self._config.num_class)
            cls_preds = torch.stack([cls_preds[i, anchor_indices] for i in range(batch_size)])
            box_preds = box_preds.view(batch_size, -1, BOX_ENCODE_SIZE)
            box_preds = torch.stack([box_preds[i, anchor_indices] for i in range(batch_size)])
            dir_targets = None
            dir_preds = None
            if self.use_direction_classifier:
                dir_targets = example["dir_cls_preds"]
                dir_preds = dir_preds.view(batch_size, -1, 2)
                dir_preds = torch.stack([dir_preds[i, anchor_indices] for i in range(batch_size)])

            cls_weights, reg_weights, dir_weights = alloc_loss_weights(
                cls_targets,
                pos_cls_weight=self._config.pos_class_weight,
                neg_cls_weight=self._config.neg_class_weight,
                dtype=voxel_data.dtype)

            loc_loss, cls_loss, dir_loss = create_loss(
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

            loc_loss_reduced = loc_loss.sum() / batch_size
            loc_loss_reduced *= self._config.loc_loss_weight
            cls_loss_reduced = cls_loss.sum() / batch_size
            cls_loss_reduced *= self._config.cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            if self.use_direction_classifier:
                dir_loss_reduced = dir_loss.sum() / batch_size
                dir_loss_reduced *= self._config.dir_loss_weight
                loss += dir_loss_reduced

            return {
                "loss": loss,
                "dir_loss_reduced": dir_loss_reduced,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "num_pos": (cls_targets > 0)[0].float().sum()
            }
        else:
            return {
                "cls_preds": cls_preds,
                "box_preds": box_preds
            }
    
def create_model(config):
    voxel_config = config.voxel_config
    x_size = math.ceil((voxel_config.x_range_max - voxel_config.x_range_min) \
                        / voxel_config.x_resolution)
    y_size = math.ceil((voxel_config.y_range_max - voxel_config.y_range_min) \
                        / voxel_config.y_resolution)
    grid_size = [x_size, y_size]

    model = PointPillars(grid_size, config)

    return model