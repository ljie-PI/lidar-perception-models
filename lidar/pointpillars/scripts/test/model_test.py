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

import pointpillars

class ModelTest(unittest.TestCase):

    def test_pointpillar_scatter(self):
        pp_scatter = pointpillars.PointPillarsScatter([5, 5], 3)
        input_feat = torch.tensor([
            [[0.01, 0.02, 0.03],
             [0.11, 0.12, 0.13],
             [0.21, 0.22, 0.23],
             [0.31, 0.32, 0.33],
             [0.41, 0.42, 0.43]],
            [[1.01, 1.02, 1.03],
             [1.11, 1.12, 1.13],
             [1.21, 1.22, 1.23],
             [1.31, 1.32, 1.33],
             [1.41, 1.42, 1.43]]
        ])
        coords = torch.tensor([
            [[1, 1, 1],
             [0, 0, 0],
             [3, 3, 3],
             [4, 4, 4],
             [2, 2, 2]],
            [[0, 2, 1],
             [2, 2, 0],
             [4, 2, 3],
             [1, 2, 4],
             [3, 2, 2]]
        ])
        output = pp_scatter(input_feat, coords, 2)
        # shape: (2, 3, 5, 5)
        gt = torch.tensor([
            [[[0.11, 0.00, 0.00, 0.00, 0.00],
              [0.00, 0.01, 0.00, 0.00, 0.00],
              [0.00, 0.00, 0.41, 0.00, 0.00],
              [0.00, 0.00, 0.00, 0.21, 0.00],
              [0.00, 0.00, 0.00, 0.00, 0.31]],
             [[0.12, 0.00, 0.00, 0.00, 0.00],
              [0.00, 0.02, 0.00, 0.00, 0.00],
              [0.00, 0.00, 0.42, 0.00, 0.00],
              [0.00, 0.00, 0.00, 0.22, 0.00],
              [0.00, 0.00, 0.00, 0.00, 0.32]],
             [[0.13, 0.00, 0.00, 0.00, 0.00],
              [0.00, 0.03, 0.00, 0.00, 0.00],
              [0.00, 0.00, 0.43, 0.00, 0.00],
              [0.00, 0.00, 0.00, 0.23, 0.00],
              [0.00, 0.00, 0.00, 0.00, 0.33]]],
            [[[0.00, 0.00, 1.01, 0.00, 0.00],
              [0.00, 0.00, 1.31, 0.00, 0.00],
              [0.00, 0.00, 1.11, 0.00, 0.00],
              [0.00, 0.00, 1.41, 0.00, 0.00],
              [0.00, 0.00, 1.21, 0.00, 0.00]],
             [[0.00, 0.00, 1.02, 0.00, 0.00],
              [0.00, 0.00, 1.32, 0.00, 0.00],
              [0.00, 0.00, 1.12, 0.00, 0.00],
              [0.00, 0.00, 1.42, 0.00, 0.00],
              [0.00, 0.00, 1.22, 0.00, 0.00]],
             [[0.00, 0.00, 1.03, 0.00, 0.00],
              [0.00, 0.00, 1.33, 0.00, 0.00],
              [0.00, 0.00, 1.13, 0.00, 0.00],
              [0.00, 0.00, 1.43, 0.00, 0.00],
              [0.00, 0.00, 1.23, 0.00, 0.00]]]
        ])
        tchtest.assert_allclose(gt, output, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
