#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import math
import unittest
import numpy as np
import numpy.testing as nptest
import torch
import torch.testing as tchtest

import box_ops

class BoxOpsTest(unittest.TestCase):

    def test_encode_box(self):
        anchor = np.array([
            [0.1, 0.2, 1.0, 2.9, 2.1, 1.2, 0.1],
            [0.2, 0.3, 1.1, 3.0, 2.2, 1.3, 0.2]
        ])
        label = np.array([
            [0.0, 0.0, 0.9, 3.0, 2.0, 1.0, 0.15],
            [0.1, 0.1, 1.0, 3.1, 2.1, 1.1, 0.25]
        ])
        encoded = np.array([
            [-0.02792904, -0.05585808, -0.16666667, 0.03390155, -0.04879016, -0.18232156, 0.05],
            [-0.02688017, -0.05376033, -0.15384615, 0.03278982, -0.04652002, -0.16705408, 0.05]
        ])
        nptest.assert_almost_equal(encoded, box_ops.encode_box(label, anchor))
        nptest.assert_almost_equal(label, box_ops.decode_box(encoded, anchor))

        anchor_tch = torch.from_numpy(anchor)
        label_tch = torch.from_numpy(label)
        encoded_tch = torch.from_numpy(encoded)
        tchtest.assert_allclose(encoded_tch, box_ops.encode_box_torch(label_tch, anchor_tch))
        tchtest.assert_allclose(label_tch, box_ops.decode_box_torch(encoded_tch, anchor_tch))

    def test_encode_direction_class(self):
        label = np.array([
            [0.0, 0.0, 0.9, 3.0, 2.0, 1.0, 0.15],
            [0.1, 0.1, 1.0, 3.1, 2.1, 1.1, -0.25]
        ])
        nptest.assert_array_equal(np.array([1, 0]), box_ops.encode_direction_class(label))
    
    def test_center_to_minmax_2d(self):
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        sizes = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        minmax = torch.tensor([
            [-0.5, -0.5, 0.5, 0.5],
            [ 0.0, 0.0, 1.0, 1.0]
        ])
        tchtest.assert_allclose(minmax, box_ops.center_to_minmax_2d(centers, sizes))
    
    def test_corner_to_standup_nd(self):
        corners = torch.tensor([[[ -0.5000, -0.5000],
                                 [ -0.5000,  0.5000],
                                 [ 0.5000,  0.5000],
                                 [ 0.5000, -0.5000]],
                                 [[-0.2071,  0.5000],
                                 [ 0.5000,  1.2071],
                                 [ 1.2071,  0.5000],
                                 [ 0.5000, -0.2071]]])
        standup_box = torch.tensor([[-0.5000, -0.5000,  0.5000,  0.5000],
                                    [-0.2071, -0.2071,  1.2071,  1.2071]])
        tchtest.assert_allclose(standup_box, box_ops.corner_to_standup_nd(corners))


if __name__ == "__main__":
    unittest.main()
