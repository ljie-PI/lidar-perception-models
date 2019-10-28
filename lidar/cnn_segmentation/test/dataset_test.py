#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import tensorflow as tf
from dataset import SparseMultiChanData, DataSet
from tfrecord_util import TFRecordUtil

class TFRecordUtilTest(tf.test.TestCase):

    def testSparseMultiChanData(self):
        indices = [(0, 0), (1, 2), (3, 1)]
        values = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        shape = (4, 4, 2)
        data = SparseMultiChanData(indices, values, shape)
        self.assertAllEqual([0, 0, 1, 1, 3, 3], data.nonempyt_height_indices())
        self.assertAllEqual([0, 0, 2, 2, 1, 1], data.nonempty_width_indices())
        self.assertAllEqual([0, 1, 0, 1, 0, 1], data.nonempty_chan_indices())
        self.assertAllEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], data.nonempty_values())
    
    def makeGoldenOutput(self):
        golden_feats = [
            [
                [1.0, 2.0, 3.0, 5.0, 6.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [9.0, 10.0, 11.0, 13.0, 14.0, 16.0]
            ]
        ]
        golden_labels = [
            [
                [0.2, 1.0, 2.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 10.0, 11.0, 12.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 13.0, 14.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 22.0, 23.0, 24.0]
            ]
        ]
        golden_example_id = [1]
        return golden_feats, golden_labels, golden_example_id
    
    def testDataSet(self):
        dataset = DataSet(
            "./test_data/feature",  # feature dir
            "./test_data/label",    # label dir
            2,  # height
            2,  # width
            6,  # in_channel
            12   # out_channel
        )
        tfrec_util = TFRecordUtil(2, 2, 6, 12)
        example = dataset.parse_example("1")
        example_ser = example.to_tfrecord()
        
        # parse example
        example_deser = tfrec_util.deserialize(example_ser)

        golden_feats, golden_labels, golden_example_id = self.makeGoldenOutput()
        
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            feature_map = tf.sparse.to_dense(example_deser[TFRecordUtil.FEATURE_MAP_KEY])
            self.assertAllEqual([2, 2, 6], feature_map.shape)
            self.assertNDArrayNear(golden_feats, sess.run(feature_map), err=0.00001)

            labels = tf.sparse.to_dense(example_deser[TFRecordUtil.LABELS_KEY])
            self.assertAllEqual([2, 2, 12], labels.shape)
            self.assertNDArrayNear(golden_labels, sess.run(labels), err=0.00001)

            example_id = example_deser[TFRecordUtil.EXAMPLE_ID_KEY]
            self.assertAllEqual(golden_example_id, sess.run(example_id))

if __name__ == "__main__":
    tf.test.main()