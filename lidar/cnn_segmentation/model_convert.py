#!/usr/bin/env python

# Convert model checkpoint files to UFF formated model

import os
import uff
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from config import ModelConfig
from cnnseg_model import CNNSegModel
from common_util import CommonUtil

class ModelLoader(object):
    def __init__(self, model):
        self._model = model
    
    def load(self, ckpt_dir):
        assignment_map = self._model.get_var_assign_map(ckpt_dir)
        tf.train.init_from_checkpoint(ckpt_dir, assignment_map)

def check_flags():
    if FLAGS.config_file is None:
        raise ValueError("No config_file is specified")
    if FLAGS.model_dir is None:
        raise ValueError("Model directory is not specified")

def main(_):
    check_flags()
    
    config = ModelConfig(FLAGS.config_file)

    # for efficiency and convenience, we use tf.data and tf.estimator APIs,
    # which need to convert examples to sparse tensors and tf.Example format.
    # the conversion will take extra effert in inference.
    # so we build another(but same structure) graph using CNNSegModel,
    # and load all the weights from checkpoint files
    with tf.Graph().as_default():
        input_ph = tf.placeholder(tf.float32,
                                  shape=(1, config.in_channel, config.height, config.width),
                                  name="data")
        mask_ph = tf.placeholder(tf.float32,
                                 shape=(1, 1, config.height, config.width),
                                 name="mask")
        input_NHWC = CommonUtil.NCHW_to_NHWC(input_ph, name="input_NHWC")
        mask_NHWC = CommonUtil.NCHW_to_NHWC(mask_ph, name="mask_NHWC")
        cnnseg_model = CNNSegModel(config, input_NHWC, None, mask_NHWC, None, False)
        instance = CommonUtil.NHWC_to_NCHW(cnnseg_model.instance,
                                           name="instance_pt")
        category_score = CommonUtil.NHWC_to_NCHW(cnnseg_model.category_score,
                                                 name="category_score")
        confidence_score = CommonUtil.NHWC_to_NCHW(cnnseg_model.confidence_score,
                                                   name="confidence_score")
        class_score = CommonUtil.NHWC_to_NCHW(cnnseg_model.class_score,
                                                 name="class_score")
        heading = CommonUtil.NHWC_to_NCHW(cnnseg_model.heading,
                                          name="heading_pt")
        height = CommonUtil.NHWC_to_NCHW(cnnseg_model.height,
                                         name="height_pt")
        
        ModelLoader(cnnseg_model).load(FLAGS.model_dir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            runtime_model_dir = os.path.join(FLAGS.model_dir, "runtime_model")
            # save to saved model
            saved_model_path = os.path.join(runtime_model_dir, "cnnseg_saved_model")
            tf.saved_model.simple_save(
                sess,
                export_dir=saved_model_path,
                inputs={"data": input_ph, "mask": mask_ph},
                outputs={
                    "instance_pt": instance,
                    "category_score": category_score,
                    "confidence_score": confidence_score,
                    "class_score": class_score,
                    "heading_pt": heading,
                    "height_pt": height
                }
            )

            # convert to .pb frozen model
            frozen_model_path = os.path.join(runtime_model_dir, "cnnseg_model.pb")
            freeze_graph.freeze_graph(
                input_graph="",
                input_saver="",
                input_binary=False,
                input_checkpoint="",
                output_node_names="instance_pt,\
                                   category_score,\
                                   confidence_score,\
                                   class_score,\
                                   heading_pt,\
                                   height_pt",
                restore_op_name="",
                filename_tensor_name="",
                output_graph=frozen_model_path,
                clear_devices=True,
                initializer_nodes="",
                input_saved_model_dir=saved_model_path,
            )

            # convert to uff formated model
            uff_model_path = os.path.join(runtime_model_dir, "cnnseg_model.uff")
            uff.from_tensorflow_frozen_model(
                frozen_file=frozen_model_path,
                output_nodes=[
                    "instance_pt",
                    "category_score",
                    "confidence_score",
                    "class_score",
                    "heading_pt",
                    "height_pt"
                ],
                output_filename=uff_model_path,
                text=False
            )

if __name__ == "__main__":
    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string("config_file", None, "Configs in json format")
    flags.DEFINE_string("model_dir", None, 
            "Model directory containing input checkpoint files and output model files")

    tf.app.run()
