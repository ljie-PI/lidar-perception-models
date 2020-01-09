#!/usr/bin/env python

import os
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from box_ops import encode_box, encode_direction_class
from generated import pp_config_pb2, pp_example_pb2


class PointPillarsDataset(Dataset):

    def __init__(self, config, data_path, is_train=True):

        self._config = config

        self._num_voxels = self._config.voxel_config.num_voxels
        self._num_points_per_voxels = self._config.voxel_config.num_points_per_voxels
        self._voxel_padding_phase = self._config.voxel_config.padding_phase

        example_ids_path = os.path.join(data_path, "example_infos.txt")
        fexample_ids = open(example_ids_path)
        self._example_ids = list(map(lambda ln: ln.strip("\r\n"), fexample_ids.readlines()))
        fexample_ids.close()

        self._data_path = data_path
        self._is_train = is_train

    def __len__(self):
        return len(self._example_ids)

    def __getitem__(self, idx):
        example_path = os.path.join(self._data_path, self._example_ids[idx] + ".bin")
        fexample = open(example_path, "rb")
        example = pp_example_pb2.Example()
        example.ParseFromString(fexample.read())
        fexample.close()

        return self._preprocess(example)

    def _preprocess(self, example):
        # parse voxel encoding
        voxel_data = np.array(list(example.voxel.data))
        voxel_data_shape = list(example.voxel.shape)
        voxel_data = voxel_data.reshape(voxel_data_shape)
        voxel_coord_data = np.array(list(example.voxel_coord.data))
        voxel_coord_data_shape = list(example.voxel_coord.shape)
        voxel_coord_data = voxel_coord_data.reshape(voxel_coord_data_shape)
        voxel_points_data = np.array(list(example.voxel_points.data))
        voxel_points_data_shape = list(example.voxel_points.shape)
        voxel_points_data = voxel_points_data.reshape(voxel_points_data_shape)

        fill_num = voxel_data_shape[0]
        pad_num = self._num_voxels - fill_num
        
        if pad_num > 0:
            pad_voxel_data = np.zeros([pad_num, self._num_points_per_voxels])
            voxel_data = np.concatenate([voxel_data, pad_voxel_data], axis=0)
            pad_voxel_coord = np.zeros([pad_num, voxel_coord_data_shape[1]])
            voxel_coord_data = np.concatenate([voxel_coord_data, pad_voxel_coord], axis=0)
            pad_voxel_points = np.zeros([pad_num])
            voxel_points_data = np.concatenate([voxel_points_data, pad_voxel_points])
        
        ret = {
            "voxel_data": voxel_data,
            "voxel_coord": voxel_coord_data,
            "voxel_points": voxel_points_data
        }
        if not self._is_train:
            return ret

        # following process are only for training
        # parse labels
        label_cnt = len(example.label)
        label_data = np.zeros([label_cnt, 7], dtype=np.float)
        label_type = np.zeros([label_cnt], dtype=np.int)
        for i in range(label_cnt):
            label = example.label[i]
            label_data[i, :] = [label.center_x, label.center_y, 0,
                                label.length, label.width, label.height, label.yaw]
            label_type[i] = label.type

        # parse anchors
        anchor_cnt = len(example.anchor)
        anchor_data = np.zeros([anchor_cnt, 8], dtype=np.float)
        anchor_targets = np.zeros([anchor_cnt], dtype=np.int)
        anchor_positive = np.zeros([anchor_cnt], dtype=np.bool)

        for i in range(anchor_cnt):
            anchor = example.anchor[i]
            anchor_data[i, :] = [anchor.offset, anchor.center_x,
                                 anchor.center_y, anchor.center_z,
                                 anchor.length, anchor.width,
                                 anchor.height, anchor.rotation]
            anchor_targets[i] = anchor.target_label
            anchor_positive[i] = anchor.is_positive
        
        sample_anchor_size = self._config.train_config.sample_anchor_size
        max_pos_anchor_size = self._config.train_config.max_pos_anchor_size
        max_neg_match_anchor_size = self._config.train_config.max_neg_match_anchor_size

        match_pos_idx = np.where(np.all(anchor_targets != -1, anchor_positive == True))[0]
        if match_pos_idx.shape[0] > max_pos_anchor_size:
            # sampling positive anchors
            match_pos_idx = np.random.permutation(match_pos_idx)[:max_pos_anchor_size]
            sample_anchor_size -= max_pos_anchor_size
        else:
            sample_anchor_size -= match_pos_idx.shape[0]

        match_neg_idx = np.where(np.all(anchor_targets != -1, anchor_positive == False))[0]
        if match_neg_idx.shape[0] > max_neg_match_anchor_size:
            # sampling matched negative anchors
            match_neg_idx = np.random.permutation(match_neg_idx)[:max_neg_match_anchor_size]
            sample_anchor_size -= max_neg_match_anchor_size
        else:
            sample_anchor_size -= match_neg_idx.shape[0]

        unmatch_idx = np.where(anchor_targets == -1)[0]
        if unmatch_idx.shape[0] > sample_anchor_size:
            # sampling match negative anchors
            unmatch_idx = np.random.permutation(unmatch_idx)[:sample_anchor_size]
        else:
            # almost impossible
            unmatch_idx = np.ones([sample_anchor_size - unmatch_idx.shape[0]]) * unmatch_idx[0]

        sampled_idx = np.concatenate([match_pos_idx, match_neg_idx, unmatch_idx])
        anchor_data = anchor_data[sampled_idx]
        anchor_targets = anchor_targets[sampled_idx]
        anchor_positive = anchor_positive[sampled_idx]
        anchor_indices = anchor_data[..., 0]

        cls_targets = label_type[anchor_targets]
        pos_anchor_mask = anchor_positive.astype(np.float)
        cls_targets = cls_targets * pos_anchor_mask

        reg_targets = encode_box(label_data[anchor_targets], anchor_data[..., 1:])

        dir_targets = None
        if self._config.train_confg.use_dir_class:
            dir_targets = encode_direction_class(label_data[anchor_targets])

        ret.update({
            "anchor_indices": anchor_indices,
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "dir_targets": dir_targets
        })
        return ret


def merge_data_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for k, v in example_merged.items():
        ret[k] = np.stack(v, axis=0)
    return ret


def create_data_loader(config, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
        num_workers=config.train_config.data_load_threads,
        pin_memory=False,
        collate_fn=merge_data_batch)
    return data_loader
