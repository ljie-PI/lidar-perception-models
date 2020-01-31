#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import unittest
import numpy as np
import numpy.testing as nptest
import torch
import torch.testing as tchtest

import loss

class LossTest(unittest.TestCase):

    def test_alloc_loss_weights(self):
        labels = torch.tensor([
            [1, 3, 4, 1, 2, 0, 0, 0, 0, 0],
            [3, 3, 5, 2, 1, 2, 4, 0, 0, 0],
            [2, 1, 2, 0, 0, 0, 2, 4, 0, 2]
        ])
        cls_weights, reg_weights, dir_weights = \
                loss.alloc_loss_weights( labels, 1.0, 1.0)
        cls_wt_gt = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285],
            [0.16666, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666]
        ])
        reg_wt_gt = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.0, 0.0, 0.0],
            [0.16666, 0.16666, 0.16666, 0.0, 0.0, 0.0, 0.16666, 0.16666, 0.0, 0.16666]
        ])
        dir_wt_gt = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.14285, 0.0, 0.0, 0.0],
            [0.16666, 0.16666, 0.16666, 0.0, 0.0, 0.0, 0.16666, 0.16666, 0.0, 0.16666]
        ])
        tchtest.assert_allclose(cls_wt_gt, cls_weights, rtol=1e-5, atol=1e-5)
        tchtest.assert_allclose(reg_wt_gt, reg_weights, rtol=1e-5, atol=1e-5)
        tchtest.assert_allclose(dir_wt_gt, dir_weights, rtol=1e-5, atol=1e-5)
    
    def test_one_hot(self):
        in_tensor = torch.tensor([
            [1, 2, 0, 1, 0],
            [1, 0, 2, 0, 2]
        ])
        one_hot = loss.one_hot(in_tensor, 3)
        one_hot_gt = torch.tensor([
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]]
        ])
        tchtest.assert_allclose(one_hot_gt, one_hot)


if __name__ == "__main__":
    unittest.main()
