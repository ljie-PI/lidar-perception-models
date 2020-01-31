#!/usr/bin/env python

import os
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from box_ops import encode_box, encode_direction_class, BOX_ENCODE_SIZE
from generated import pp_config_pb2, pp_example_pb2


class PointPillarsDataset(Dataset):

    def __init__(self, config, data_path, is_train=True):

        self._config = config

        self._num_voxels = self._config.voxel_config.num_voxels
        self._num_points_per_voxel = self._config.voxel_config.num_points_per_voxel

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

        return self.preprocess(example, int(self._example_ids[idx]))

    def preprocess(self, example, example_id):
        # parse voxel encoding
        voxel_coord_data = np.array(list(example.voxel_coord.data))
        voxel_coord_data = voxel_coord_data.reshape(
            [example.voxel_coord.num_voxel, example.voxel_coord.coord_dim])
        voxel_points_data = np.array(list(example.voxel_points.data), dtype=np.int)
        voxel_points_data = voxel_points_data.reshape([example.voxel_points.num_voxel])
        num_voxel = example.voxel.num_voxel
        feature_dim = example.voxel.feature_dim
        assert num_voxel == example.voxel_coord.num_voxel \
           and num_voxel == example.voxel_points.num_voxel
        voxel_comp_data = np.array(list(example.voxel.data)).reshape([-1, feature_dim])
        voxel_data = np.zeros([num_voxel, self._num_points_per_voxel, feature_dim])
        offset = 0
        for vidx in range(voxel_points_data.shape[0]):
            num_points = voxel_points_data[vidx]
            voxel_data[vidx, :num_points] = voxel_comp_data[offset:offset+num_points]
            offset += num_points

        # do padding
        pad_num = self._num_voxels - num_voxel
        if pad_num > 0:
            pad_voxel_data = np.zeros([pad_num, self._num_points_per_voxel, feature_dim])
            voxel_data = np.concatenate([voxel_data, pad_voxel_data], axis=0)
            pad_voxel_coord = np.zeros([pad_num, example.voxel_coord.coord_dim])
            voxel_coord_data = np.concatenate([voxel_coord_data, pad_voxel_coord], axis=0)
            pad_voxel_points = np.zeros([pad_num])
            voxel_points_data = np.concatenate([voxel_points_data, pad_voxel_points])
        
        ret = {
            "voxel_data": voxel_data,
            "voxel_coord": voxel_coord_data,
            "voxel_points": voxel_points_data
        }
        if not self._is_train:
            ret.update({
                "example_id": np.array([example_id], dtype=np.int32)
            })
            return ret

        # following process are only for training
        # parse labels
        label_cnt = len(example.label)
        label_data = np.zeros([label_cnt, 7], dtype=np.float)
        label_type = np.zeros([label_cnt], dtype=np.int)
        for i in range(label_cnt):
            label = example.label[i]
            label_data[i, :] = [label.center_x, label.center_y, label.center_z,
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

        match_pos_idx = np.where(np.logical_and(anchor_targets != -1, anchor_positive == True))[0]
        if match_pos_idx.shape[0] > max_pos_anchor_size:
            # sampling positive anchors
            match_pos_idx = np.random.permutation(match_pos_idx)[:max_pos_anchor_size]
            sample_anchor_size -= max_pos_anchor_size
        else:
            sample_anchor_size -= match_pos_idx.shape[0]

        match_neg_idx = np.where(np.logical_and(anchor_targets != -1, anchor_positive == False))[0]
        if match_neg_idx.shape[0] > max_neg_match_anchor_size:
            # sampling matched negative anchors
            match_neg_idx = np.random.permutation(match_neg_idx)[:max_neg_match_anchor_size]
            sample_anchor_size -= max_neg_match_anchor_size
        else:
            sample_anchor_size -= match_neg_idx.shape[0]

        sampled_idx = np.concatenate([match_pos_idx, match_neg_idx])
        m_anchor_data = anchor_data[sampled_idx]
        m_anchor_targets = anchor_targets[sampled_idx]
        m_anchor_positive = anchor_positive[sampled_idx]
        m_anchor_indices = m_anchor_data[..., 0]

        m_cls_targets = label_type[m_anchor_targets]
        pos_anchor_mask = m_anchor_positive.astype(np.float)
        m_cls_targets = m_cls_targets * pos_anchor_mask
        m_reg_targets = encode_box(label_data[m_anchor_targets], m_anchor_data[..., 1:])
        m_dir_targets = None
        if self._config.model_config.use_dir_class:
            m_dir_targets = encode_direction_class(label_data[m_anchor_targets])

        unmatch_idx = np.where(anchor_targets == -1)[0]
        if unmatch_idx.shape[0] > sample_anchor_size:
            # sampling match negative anchors
            unmatch_idx = np.random.permutation(unmatch_idx)[:sample_anchor_size]
        else:
            # almost impossible, duplicate the first one
            more_unmatch_idx = np.ones([sample_anchor_size - unmatch_idx.shape[0]], dtype=np.int32) * unmatch_idx[0]
            unmatch_idx = np.concatenate([unmatch_idx, more_unmatch_idx], axis=0)
        um_anchor_data = anchor_data[unmatch_idx]
        um_anchor_indices = um_anchor_data[..., 0]
        um_cls_targets = np.zeros([unmatch_idx.shape[0]])
        um_reg_targets = np.zeros([unmatch_idx.shape[0], BOX_ENCODE_SIZE])
        um_dir_targets = None
        if self._config.model_config.use_dir_class:
            um_dir_targets = np.zeros([unmatch_idx.shape[0]])
        
        anchor_indices = np.concatenate([m_anchor_indices, um_anchor_indices], axis=0)
        cls_targets = np.concatenate([m_cls_targets, um_cls_targets], axis=0)
        reg_targets = np.concatenate([m_reg_targets, um_reg_targets], axis=0)
        dir_targets = None
        if self._config.model_config.use_dir_class:
            dir_targets = np.concatenate([m_dir_targets, um_dir_targets], axis=0)

        ret.update({
            "anchor_indices": anchor_indices.astype(np.int32),
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "dir_targets": dir_targets
        })

        return ret


def merge_data_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    none_keys = set()
    for example in batch_list:
        for k, v in example.items():
            if v is None:
                none_keys.add(k)
                continue
            example_merged[k].append(v)
    ret = {}
    for k, v in example_merged.items():
        ret[k] = np.stack(v, axis=0)
    for k in none_keys:
        ret[k] = None
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

def create_eval_data_loader(config, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=config.eval_config.batch_size,
        shuffle=False,
        num_workers=config.eval_config.data_load_threads,
        pin_memory=False,
        collate_fn=merge_data_batch)
    return data_loader
