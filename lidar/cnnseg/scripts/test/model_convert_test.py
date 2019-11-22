#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import time
import collections

import tensorflow as tf
import numpy as np

from mock_model import MockModel
from model_convert import ModelLoader
from common_util import CommonUtil

class ModelConvertTest(tf.test.TestCase):

    def set_model_dir(self):
        ts = int(time.time())
        self.model_dir = "/tmp/model_convert_test_{:d}".format(ts)
        os.mkdir(self.model_dir)

    def train_model(self):
        def model_fn(features, labels, mode, params):
            features = tf.reshape(features, [-1, 4, 4, 1])
            if labels is not None:
                labels = tf.to_int32(labels)
                labels = tf.one_hot(
                    indices=labels,
                    depth=8
                )
                labels = tf.reshape(labels, [-1, 8])
            model = MockModel(features, labels, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdadeltaOptimizer(0.0005)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
                with tf.control_dependencies(update_ops):
                    grads_and_vars = optimizer.compute_gradients(model.loss)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=model.loss,
                    train_op=train_op
                )
            elif mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {"output": model.output}
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions
                )
            return output_spec
        
        def parse_examples(records):
            dec_tensor = tf.decode_csv(
                records=records,
                record_defaults=[[0.0] for i in range(17)],
                field_delim=" "
            )
            dec_tensor = tf.reshape(dec_tensor, shape=[17])
            labels, features = tf.split(dec_tensor, [1, 16], axis=-1)
            return features, labels
        
        def train_input_fn():
            data = tf.data.TextLineDataset("./test_data/model_convert/train_examples")
            data = data.map(lambda rec: parse_examples(rec))
            data = data.repeat().shuffle(buffer_size=8).batch(8)
            return data

        def pred_input_fn():
            data = tf.data.TextLineDataset("./test_data/model_convert/pred_examples")
            data = data.map(lambda rec: parse_examples(rec))
            return data

        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=self.model_dir)
        estimator.train(input_fn=train_input_fn, max_steps=100)
        pred_res = estimator.predict(input_fn=pred_input_fn)
        self.golden_out = []
        for (_, prediction) in enumerate(pred_res):
            self.golden_out.append(prediction["output"])
    
    def load_model_and_preidict(self):
        def load_pred_examples():
            pred_in = []
            fpred = open("./test_data/model_convert/pred_examples")
            for lines in fpred.readlines():
                feats = lines.strip().split(" ")[1:]
                feat_reshape = np.array(list(map(lambda  x: float(x), feats))).reshape([1, 1, 4, 4])
                pred_in.append(feat_reshape)
            return pred_in

        with tf.Graph().as_default():
            input_ph = tf.placeholder(tf.float32, shape=[1, 1, 4, 4], name="input_feature")
            input_NHWC = CommonUtil.NCHW_to_NHWC(input_ph, name="input_NHWC")
            model = MockModel(input_NHWC, None, False)
            output = model.output
            ModelLoader(model).load(self.model_dir)

            pred_in = load_pred_examples()
            self.pred_out = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for in_example in pred_in:
                    self.pred_out.append(sess.run(output, feed_dict={input_ph: in_example}))

    def testModelSaveLoad(self):
        print("==================== test model save and load ====================")

        self.set_model_dir()
        print("==================== model dir is: {} ====================".format(self.model_dir))

        self.train_model()
        print("==================== model training finished ====================")
        print("==================== old model predictions: ====================")
        print(self.golden_out)

        self.load_model_and_preidict()
        print("==================== loaded pre-trained model ====================")
        print("==================== new model predictions: ====================")
        print(self.pred_out)
        
        self.assertNDArrayNear(np.array(self.golden_out), np.array(self.pred_out), err=0.00001)


if __name__ == "__main__":
    tf.test.main()