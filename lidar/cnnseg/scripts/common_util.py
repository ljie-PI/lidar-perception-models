#!/usr/bin/env python

import tensorflow as tf

class CommonUtil(object):

    @staticmethod
    def random_set(shape, ratio, value, set_area=None, rand_seed=None):
        """
        random set elements to `value` in tensor(shaped `shape`) with `ratio` in `set_area`
        `set_area` is also in shape `shape`. values in `set_area` are either 0 or 1,
        1 means that element is possible to be set to `value`
        0 means that element can not be set to `value`
        if `set_area` is None, all elements is possible to be set to `value`
        """
        uniform_random = tf.random.uniform(shape, minval=0, maxval=1)
        rand_area = tf.to_float(tf.less(uniform_random, ratio))
        set_result = tf.multiply(rand_area, value)
        if set_area is None:
            return set_result
        return tf.multiply(set_result, set_area)

    @staticmethod
    def NHWC_to_NCHW(tensor, name=None):
        if name is None:
            return tf.transpose(tensor, [0, 3, 1, 2])
        return tf.transpose(tensor, [0, 3, 1, 2], name=name)

    @staticmethod
    def NCHW_to_NHWC(tensor, name=None):
        if name is None:
            return tf.transpose(tensor, [0, 2, 3, 1])
        return tf.transpose(tensor, [0, 2, 3, 1], name=name)