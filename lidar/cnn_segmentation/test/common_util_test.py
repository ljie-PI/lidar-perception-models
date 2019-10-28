#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import tensorflow as tf
import numpy as np
from common_util import CommonUtil


class TFRecordUtilTest(tf.test.TestCase):
    def testRandomSet(self):
        set_area = np.array(
        [
            [
                [0, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 0, 1]
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 1]
            ]
        ])
        set_area_ph = tf.placeholder(dtype=tf.float32, shape=[2, 3, 4])
        rand_set = CommonUtil.random_set(
            shape=tf.shape(set_area_ph),
            ratio=0.5,
            value=3.0,
            set_area=set_area_ph,
            rand_seed=1
        )
        golden_out = np.array(
        [
            [
                [0, 0, 0, 0],
                [3, 0, 0, 0],
                [0, 3, 0, 3]
            ],
            [
                [0, 3, 0, 0],
                [0, 0, 3, 0],
                [3, 0, 0, 0]
            ]
        ])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            rand_set_result = sess.run(rand_set, feed_dict={set_area_ph: set_area})
            self.assertAllEqual(golden_out, rand_set_result)
            

if __name__ == "__main__":
    tf.test.main()
