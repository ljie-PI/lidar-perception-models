#!/usr/bin/env python

import numpy as np
import torch

BOX_ENCODE_SIZE = 7

def encode_box(boxes, anchors):
    xa, ya, za, la, wa, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, lg, wg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    lt = np.log(lg / la)
    wt = np.log(wg / wa)
    ht = np.log(hg / ha)
    rt = rg - ra
    return np.concatenate([xt, yt, zt, lt, wt, ht, rt], axis=-1)


def encode_box_torch(boxes, anchors):
    xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, lg, wg, hg, rg = torch.split(boxes, 1, dim=-1)
    za = za + ha / 2
    zg = zg + hg / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    lt = torch.log(lg / la)
    wt = torch.log(wg / wa)
    ht = torch.log(hg / ha)
    rt = rg - ra
    return torch.cat([xt, yt, zt, lt, wt, ht, rt], dim=-1)


def decode_box(box_encodings, anchors):
    xa, ya, za, la, wa, ha, ra = np.split(anchors, 7, axis=-1)
    xt, yt, zt, lt, wt, ht, rt = np.split(box_encodings, 7, axis=-1)
    za = za + ha / 2
    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = np.exp(lt) * la
    wg = np.exp(wt) * wa
    hg = np.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    return np.concatenate([xg, yg, zg, lg, wg, hg, rg], axis=-1)


def decode_box_torch(box_encodings, anchors):
    xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, lt, wt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, lg, wg, hg, rg], dim=-1)


def encode_direction_class(label_data):
    rot = label_data[..., -1]
    dir_cls_targets = (rot > 0).astype(np.int32)
    return dir_cls_targets


def center_to_minmax_2d(centers, dims, origin=0.5):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d_torch(centers, dims, origin=0.5):
    return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and origin point. 
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.
    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(-angles)
    rot_cos = torch.cos(-angles)
    rot_mat_T = torch.stack(
        [torch.stack([rot_cos, -rot_sin]),
         torch.stack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)
