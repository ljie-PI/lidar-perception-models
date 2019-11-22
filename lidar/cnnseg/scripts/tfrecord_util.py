#!/usr/bin/env python

import collections
import tensorflow as tf

class TFRecordUtil(object):
    EXAMPLE_ID_KEY = "example_id"
    FEATURE_HEIGHT_IDX_KEY = "feature_height_idx"
    FEATURE_WIDTH_IDX_KEY = "feature_width_idx"
    FEATURE_CHAN_IDX_KEY = "feature_chan_idx"
    FEATURE_VALUE_KEY = "feature_value"
    FEATURE_MAP_KEY = "feature_map"
    LABEL_HEIGHT_IDX_KEY = "label_height_idx"
    LABEL_WIDTH_IDX_KEY = "label_width_idx"
    LABEL_CHAN_IDX_KEY = "label_chan_idx"
    LABEL_VALUE_KEY = "label_value"
    LABELS_KEY = "labels"

    def __init__(self, height, width, feat_channel, label_channel):
        self._feat_shapes = [height, width, feat_channel]
        self._label_shapes = [height, width, label_channel]
        self._name_to_features = {
            self.FEATURE_MAP_KEY: tf.io.SparseFeature(
                index_key=[self.FEATURE_HEIGHT_IDX_KEY,
                           self.FEATURE_WIDTH_IDX_KEY,
                           self.FEATURE_CHAN_IDX_KEY],
                value_key=self.FEATURE_VALUE_KEY,
                dtype=tf.float32,
                size=self._feat_shapes),
            self.LABELS_KEY: tf.io.SparseFeature(
                index_key=[self.LABEL_HEIGHT_IDX_KEY,
                           self.LABEL_WIDTH_IDX_KEY,
                           self.LABEL_CHAN_IDX_KEY],
                value_key=self.LABEL_VALUE_KEY,
                dtype=tf.float32,
                size=self._label_shapes),
            self.EXAMPLE_ID_KEY: tf.io.FixedLenFeature([1], tf.int64)
        }

    def serialize(self, feat_height_indices, feat_width_indices, feat_chan_indices, feat_values,
                  label_height_indices, label_width_indices, label_chan_indices, label_values, example_id):
        features = collections.OrderedDict()
        features[self.EXAMPLE_ID_KEY] = self._create_int_feature([example_id])
        features[self.FEATURE_HEIGHT_IDX_KEY] = self._create_int_feature(feat_height_indices)
        features[self.FEATURE_WIDTH_IDX_KEY] = self._create_int_feature(feat_width_indices)
        features[self.FEATURE_CHAN_IDX_KEY] = self._create_int_feature(feat_chan_indices)
        features[self.FEATURE_VALUE_KEY] = self._create_float_feature(feat_values)
        features[self.LABEL_HEIGHT_IDX_KEY] = self._create_int_feature(label_height_indices)
        features[self.LABEL_WIDTH_IDX_KEY] = self._create_int_feature(label_width_indices)
        features[self.LABEL_CHAN_IDX_KEY] = self._create_int_feature(label_chan_indices)
        features[self.LABEL_VALUE_KEY] = self._create_float_feature(label_values)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example.SerializeToString()        
    
    def deserialize(self, record):
        example = tf.io.parse_single_example(record, self._name_to_features)
        return example

    def deserialize_batch(self, records):
        example = tf.io.parse_example(records, self._name_to_features)
        return example

    def _create_int_feature(self, values):
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feature

    def _create_float_feature(self, values):
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feature