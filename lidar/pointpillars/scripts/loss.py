#!/usr/bin/env python

import logging
import torch

from box_ops import BOX_ENCODE_SIZE


def alloc_loss_weights(labels, pos_cls_weight=1.0,
                       neg_cls_weight=1.0, dtype=torch.float32):
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    dir_weights = positives.type(dtype)

    pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
    reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    dir_weights /= torch.clamp(pos_normalizer, min=1.0)

    return cls_weights, reg_weights, dir_weights


def create_loss(box_preds, cls_preds, dir_preds,
                cls_targets, cls_weights,
                reg_targets, reg_weights,
                dir_targets, dir_weights, config):
    batch_size = int(box_preds.shape[0])
    num_class = config.model_config.num_class

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    one_hot_targets = one_hot_targets[..., 1:]

    cls_losses = focal_binary_cross_entropy(
        cls_preds, one_hot_targets, config.train_config)
    cls_losses = cls_losses * cls_weights.unsqueeze(-1)

    box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)

    reg_losses = smooth_l1_loss(
        box_preds, reg_targets, config.train_config)
    reg_losses = reg_losses * reg_weights.unsqueeze(-1)

    dir_losses = None
    if config.model_config.use_dir_class:
        dir_one_hot_targets = one_hot(dir_targets, 2, dtype=dir_preds.dtype)
        dir_losses = sigmoid_cross_entropy_with_logits(
            logits=dir_preds,
            labels=dir_one_hot_targets)
        dir_losses = dir_losses * dir_weights.unsqueeze(-1)

    return reg_losses, cls_losses, dir_losses


def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def smooth_l1_loss(predict_tensor, target_tensor, train_config):
    diff = predict_tensor - target_tensor
    abs_diff = torch.abs(diff)
    sigma = train_config.smooth_l1_sigma
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) \
           + (abs_diff - 0.5 / (sigma**2)) * (1. - abs_diff_lt_1)
    return loss


def focal_binary_cross_entropy(predict_tensor, target_tensor, train_config):
    per_entry_cross_ent = sigmoid_cross_entropy_with_logits(
        logits=predict_tensor,
        labels=target_tensor)
    prediction_probabilities = torch.sigmoid(predict_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = torch.pow(1.0 - p_t, train_config.focal_gamma)
    alpha_weight_factor = (target_tensor * train_config.focal_alpha +
                          (1 - target_tensor) * (1 - train_config.focal_alpha))
    loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
    return loss


def sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss