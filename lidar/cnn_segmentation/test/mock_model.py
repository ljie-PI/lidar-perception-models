#!/usr/bin/env python

import tensorflow as tf

class MockModel(object):
    VARIABLE_SCOPE = "mock_model"

    def __init__(self, inputs, labels, is_training):
        self._is_training = is_training
        with tf.variable_scope(self.VARIABLE_SCOPE):
            self._build_graph(inputs)
            if is_training:
                self._compute_loss(labels)

    def _build_graph(self, inputs):
        filter_weight = tf.get_variable(
            "filter_weight",
            shape=[3, 3, 1, 2],
            # initializer=tf.glorot_uniform_initializer()
            initializer=tf.initializers.he_uniform()
        )
        bias = tf.get_variable(
            "bias",
            shape=[2],
            initializer=tf.zeros_initializer()
        )
        conv = tf.nn.conv2d(
            inputs,
            filter=filter_weight,
            strides=1,
            padding="VALID",
        )
        bias_add = tf.nn.bias_add(conv, bias)
        batch_norm = tf.layers.batch_normalization(
            bias_add,
            training=self._is_training
        )
        activation = tf.nn.relu(batch_norm)
        self.logits = tf.reshape(activation, [-1, 8])
        self.output = tf.nn.softmax(self.logits, name="output")

    def _compute_loss(self, labels):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=self.logits
        )
        self.loss = tf.reduce_mean(losses)
    
    def get_var_assign_map(self, ckpt_dir):
        return {self.VARIABLE_SCOPE+"/": self.VARIABLE_SCOPE+"/"}