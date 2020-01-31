#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import unittest
import numpy as np
import numpy.testing as nptest

from google.protobuf import text_format
from dataset import PointPillarsDataset, merge_data_batch
from box_ops import encode_box
from generated import pp_config_pb2

class DatasetTest(unittest.TestCase):

    def setUp(self):
        config = pp_config_pb2.PointPillarsConfig()
        fconfig = open("./test/test_data/test_config.pb.txt")
        text_format.Parse(fconfig.read(), config)
        fconfig.close()
        self.dataset = PointPillarsDataset(config, "./test/test_data", True)
        self.assertEqual(1, len(self.dataset))

    def test_voxel_data(self):
        example = self.dataset[0]
        keys = {"voxel_data", "voxel_coord", "voxel_points"}
        for key in keys:
            self.assertTrue(key in example)
        
        voxel_points = example["voxel_points"]
        self.assertTrue(9, voxel_points.shape[0])
        point_nums_gt = np.array([10, 6, 6, 6, 5, 6, 6, 6, 6, 0])
        nptest.assert_array_equal(point_nums_gt, voxel_points)

        voxel_coord = example["voxel_coord"]
        self.assertTrue(9, voxel_coord.shape[0])
        self.assertTrue(3, voxel_coord.shape[1])
        coord_gt = np.array([
            [300, 302, 0], [301, 302, 0], [301, 303, 0],
            [302, 303, 0], [304, 302, 0], [305, 302, 0],
            [306, 302, 0], [304, 303, 0], [305, 303, 0], [0, 0, 0]
        ])
        nptest.assert_array_equal(coord_gt, voxel_coord)

        voxel_data = example["voxel_data"]
        self.assertEqual(10, voxel_data.shape[0])
        self.assertEqual(10, voxel_data.shape[1])
        self.assertEqual(7, voxel_data.shape[2])
        # the 1st voxel contains $num_points_per_voxel(=10) points
        for i in range(10):
            for j in range(7):
                self.assertTrue(voxel_data[0, i, j] != 0)
        # check the 4th voxel as in voxel_generator_test.cpp
        point_4_1_gt = np.array([0.21, 0.31, 0.1, 0.37443, -0.04, -0.04, -0.9])
        nptest.assert_allclose(point_4_1_gt, voxel_data[3, 0, :],
                               rtol=1e-5, atol=1e-5)
        point_4_6_gt = np.array([0.23, 0.30, 1.3, 0.37802, -0.02, -0.05, 0.3])
        nptest.assert_allclose(point_4_6_gt, voxel_data[3, 5, :],
                               rtol=1e-5, atol=1e-5)
        # 6 non-empty points, 4 paddings
        paddings = np.zeros([4, 7])
        nptest.assert_array_equal(paddings, voxel_data[3, 6:, :])

        padding_voxel = np.zeros([1, 10, 7])
        nptest.assert_array_equal(padding_voxel, voxel_data[-1:, ...])

    def test_target_assian(self):
        example = self.dataset[0]
        keys = {"anchor_indices", "cls_targets", "reg_targets", "dir_targets"}
        for key in keys:
            self.assertTrue(key in example)
        anchor_indices = example["anchor_indices"]
        cls_targets = example["cls_targets"]
        reg_targets = example["reg_targets"]
        dir_targets = example["dir_targets"]
        pos_idx = np.where(cls_targets > 0)
        pos_anchor_indices = anchor_indices[pos_idx]
        pos_anchor_indices_gt = np.array([733211, 733214, 733215, 723610])
        nptest.assert_array_equal(pos_anchor_indices_gt, pos_anchor_indices)

        cls_targets = cls_targets[pos_idx]
        cls_targets_gt = np.array([4, 4, 4, 1])
        nptest.assert_array_equal(cls_targets_gt, cls_targets)

        reg_targets = reg_targets[pos_idx]
        anchor_data = np.array([
            [0.550001, 0.25, 0, 0.3, 0.2, 2, 1.5708],
            [0.550001, 0.35, 0, 0.3, 0.2, 2, 0],
            [0.550001, 0.35, 0, 0.3, 0.2, 2, 1.5708],
            [0.15, 0.25, 0, 0.3, 0.2, 2, 0]
        ])
        label_data = np.array([
            [0.55, 0.3, 1.0, 0.3, 0.2, 1, 1],
            [0.55, 0.3, 1.0, 0.3, 0.2, 1, 1],
            [0.55, 0.3, 1.0, 0.3, 0.2, 1, 1],
            [0.15, 0.3, 1.2, 0.2, 0.2, 1.2, 1.2]
        ])
        reg_targets_gt = encode_box(label_data, anchor_data)
        nptest.assert_allclose(reg_targets_gt, reg_targets,
                               rtol=1e-5, atol=1e-5)

        dir_targets = dir_targets[pos_idx]
        dir_targets_gt = np.array([1, 1, 1, 1])
        nptest.assert_array_equal(dir_targets_gt, dir_targets)


if __name__ == "__main__":
    unittest.main()