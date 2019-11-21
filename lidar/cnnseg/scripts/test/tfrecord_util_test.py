#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import tensorflow as tf
from tfrecord_util import TFRecordUtil


class TFRecordUtilTest(tf.test.TestCase):

    def makeSerializedExample(self, tfrec_util):
        feat_height_idx = (0, 1, 2, 2)
        feat_width_idx = (0, 1, 0, 2)
        feat_chan_idx = (1, 0, 1, 0)
        feat_value = (1.0, 2.0, 3.0, 4.0)
        label_height_idx = (0, 0, 1, 2)
        label_width_idx = (0, 1, 2, 2)
        label_chan_idx = (0, 1, 2, 3)
        label_value = (1.0, 2.0, 3.0, 4.0)
        example_id = 1
        example_ser = tfrec_util.serialize(
            feat_height_idx, feat_width_idx, feat_chan_idx, feat_value,
            label_height_idx, label_width_idx, label_chan_idx, label_value, example_id
        )
        return example_ser
    
    def makeGoldenOutput(self):
        golden_feats = [
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 3.0],
                [0.0, 0.0],
                [4.0, 0.0],
            ]
        ]
        golden_labels = [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ]
        ]
        golden_example_id = [1]
        return golden_feats, golden_labels, golden_example_id

    def makeSerializedExample2(self, tfrec_util):
        feat_height_idx = (0, 1, 2, 2)
        feat_width_idx = (0, 1, 0, 2)
        feat_chan_idx = (1, 0, 1, 0)
        feat_value = (2.0, 4.0, 6.0, 8.0)
        label_height_idx = (0, 0, 1, 2)
        label_width_idx = (0, 1, 2, 2)
        label_chan_idx = (0, 1, 2, 3)
        label_value = (2.0, 4.0, 6.0, 8.0)
        example_id = 2
        example_ser = tfrec_util.serialize(
            feat_height_idx, feat_width_idx, feat_chan_idx, feat_value,
            label_height_idx, label_width_idx, label_chan_idx, label_value, example_id
        )
        return example_ser
    
    def makeGoldenBatchOutput(self):
        golden_feats = [
            [
                [
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 3.0],
                    [0.0, 0.0],
                    [4.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [4.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 6.0],
                    [0.0, 0.0],
                    [8.0, 0.0],
                ]
            ]
        ]
        golden_labels = [
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            ],
            [
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 8.0],
                ]
            ],
        ]
        golden_example_ids = [[1], [2]]
        return golden_feats, golden_labels, golden_example_ids

    def testSerDeser(self):
        tfrec_util = TFRecordUtil(3, 3, 2, 4)
        example_ser = self.makeSerializedExample(tfrec_util)
        example_deser = tfrec_util.deserialize(example_ser)

        golden_feats, golden_labels, golden_example_id = self.makeGoldenOutput()
        
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            feature_map = tf.sparse.to_dense(example_deser[TFRecordUtil.FEATURE_MAP_KEY])
            self.assertAllEqual([3, 3, 2], feature_map.shape)
            self.assertAllEqual(golden_feats, sess.run(feature_map))

            labels = tf.sparse.to_dense(example_deser[TFRecordUtil.LABELS_KEY])
            self.assertAllEqual([3, 3, 4], labels.shape)
            self.assertAllEqual(golden_labels, sess.run(labels))

            example_id = example_deser[TFRecordUtil.EXAMPLE_ID_KEY]
            self.assertAllEqual(golden_example_id, sess.run(example_id))

    def testDeserBatch(self):
        tfrec_util = TFRecordUtil(3, 3, 2, 4)
        example_ser = self.makeSerializedExample(tfrec_util)
        example_ser2 = self.makeSerializedExample2(tfrec_util)
        input_example = tf.placeholder(dtype=tf.string, shape=[None], name="input_example_tensor")
        example_deser = tfrec_util.deserialize_batch(input_example)
        input_tensor_feed_dict = {input_example: [example_ser, example_ser2]}

        golden_feats, golden_labels, golden_example_ids = self.makeGoldenBatchOutput()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            features_maps = tf.sparse.to_dense(example_deser[TFRecordUtil.FEATURE_MAP_KEY])
            self.assertAllEqual(golden_feats, sess.run(features_maps, feed_dict=input_tensor_feed_dict))

            labels = tf.sparse.to_dense(example_deser[TFRecordUtil.LABELS_KEY])
            self.assertAllEqual(golden_labels, sess.run(labels, feed_dict=input_tensor_feed_dict))

            example_ids = example_deser[TFRecordUtil.EXAMPLE_ID_KEY]
            self.assertAllEqual(golden_example_ids, sess.run(example_ids, feed_dict=input_tensor_feed_dict))

if __name__ == "__main__":
    tf.test.main()