#!/usr/bin/env python

import os
import tensorflow as tf

from config import ModelConfig
from tfrecord_util import TFRecordUtil
from cnnseg_model import CNNSegModel

def check_flags():
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of do_train, do_eval or do_predict must be True")
    if FLAGS.do_train and FLAGS.train_inputs is None:
        raise ValueError("No training inputs specified")
    if FLAGS.do_eval and FLAGS.eval_inputs is None:
        raise ValueError("No evluating inputs specified")
    if FLAGS.do_predict and FLAGS.predict_inputs is None:
        raise ValueError("No prediction inputs specified")
    if FLAGS.model_dir is None:
        raise ValueError("No model_dir is specified")
    if FLAGS.do_predict and FLAGS.output_dir is None:
        raise ValueError("No output_dir is specified for prediction task")
    if FLAGS.config_file is None:
        raise ValueError("No config_file is specified")

def copy_config(config_file, model_dir):
    target_file = os.path.join(model_dir, "config.json")
    if os.path.exists(target_file):
        return
    os.system("cp {} {}".format(config_file, target_file))

def build_model_fn(config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        example_ids = features[TFRecordUtil.EXAMPLE_ID_KEY]
        feature_maps = tf.sparse.to_dense(features[TFRecordUtil.FEATURE_MAP_KEY])
        labels = tf.sparse.to_dense(features[TFRecordUtil.LABELS_KEY], validate_indices=False)
        feature_maps = tf.reshape(feature_maps, [-1, config.height, config.width, config.in_channel])
        labels = tf.reshape(labels, [-1, config.height, config.width, config.out_channel])
        example_ids = tf.reshape(features[TFRecordUtil.EXAMPLE_ID_KEY], [-1, 1])
        mask = tf.gather(feature_maps, tf.constant([config.mask_chan_idx]), axis=-1)
        cnnseg_model = CNNSegModel(config, feature_maps, labels, mask, example_ids,
                                   is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    
        # in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdadeltaOptimizer(config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
            with tf.control_dependencies(update_ops):
                if config.merge_loss:
                    ## use single train_op to minimize weighted-sum loss
                    grads_and_vars = optimizer.compute_gradients(cnnseg_model.loss)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
                    optimizer.minimize(cnnseg_model.loss)
                    if config.enable_summary:
                        add_grad_summaries(grads_and_vars)
                else:
                    ## use multiple train ops for every loss, and minimize separately
                    category_train_op = optimizer.minimize(cnnseg_model.category_score_loss,
                                                        global_step=tf.train.get_global_step())
                    instance_train_op = optimizer.minimize(cnnseg_model.instance_loss,
                                                        global_step=tf.train.get_global_step())
                    confidence_train_op = optimizer.minimize(cnnseg_model.confidence_score_loss,
                                                            global_step=tf.train.get_global_step())
                    classify_train_op = optimizer.minimize(cnnseg_model.class_score_loss,
                                                        global_step=tf.train.get_global_step())
                    heading_train_op = optimizer.minimize(cnnseg_model.heading_loss,
                                                        global_step=tf.train.get_global_step())
                    height_train_op = optimizer.minimize(cnnseg_model.height_loss,
                                                        global_step=tf.train.get_global_step())
                    train_op = tf.group(category_train_op, instance_train_op, confidence_train_op,
                                        classify_train_op, heading_train_op, height_train_op)
            multi_task_losses = {
                "category_loss": cnnseg_model.category_score_loss,
                "instance_loss": cnnseg_model.instance_loss,
                "confidence_loss": cnnseg_model.confidence_score_loss,
                "classify_loss": cnnseg_model.class_score_loss,
                "heading_loss": cnnseg_model.heading_loss,
                "height_loss": cnnseg_model.height_loss
            }
            logging_hook = tf.train.LoggingTensorHook(
                multi_task_losses,
                every_n_iter=config.save_summary_steps
            )
            training_hooks = [logging_hook]
            if config.enable_summary:
                summary_hook = tf.train.SummarySaverHook(
                    save_steps=config.save_summary_steps,
                    summary_op=tf.summary.merge_all()
                )
                training_hooks.append(summary_hook)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cnnseg_model.loss,
                train_op=train_op,
                training_hooks=training_hooks
            )
        # in evaluating mode
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = {
                "category_precision": cnnseg_model.category_precision,
                "category_recall": cnnseg_model.category_recall,
                "instance_rmse": cnnseg_model.instance_rmse,
                "confidence_precision": cnnseg_model.confidence_precision,
                "confidence_recall": cnnseg_model.confidence_recall,
                "classify_iou": cnnseg_model.classify_iou,
                "heading_rmse": cnnseg_model.heading_rmse,
                "height_rmse": cnnseg_model.height_rmse
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cnnseg_model.loss,
                eval_metric_ops=eval_metrics
            )
        # in prediction mode
        else:
            predictions = {
                "example_id": example_ids,
                "category_score": cnnseg_model.category_score,
                "instance": cnnseg_model.instance,
                "confidence_score": cnnseg_model.confidence_score,
                "class_score": cnnseg_model.class_score,
                "heading": cnnseg_model.heading,
                "height": cnnseg_model.height,
                "mask": mask
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        return output_spec
    
    def add_grad_summaries(grads_and_vars):
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram("{}_hist".format(var.name), var)
                tf.summary.histogram("{}_hist".format(grad.name), grad)
    
    return model_fn

def build_input_fn(input_files, config, is_train=False):
    file_list = []
    tf.logging.info("***** Input Files ({}) *****".format(input_files))
    for pattern in input_files.split(","):
        file_list.extend(tf.gfile.Glob(pattern))
    tf.logging.info("***** Input Files ({:d} files) *****".format(len(file_list)))
    for input_file in file_list:
        tf.logging.info("  {}".format(input_file))
    tfrec_util = TFRecordUtil(config.height, config.width, config.in_channel, config.out_channel)

    def input_fn():
        if is_train:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(file_list))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(file_list))
            d = d.interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=config.thread_num
            )
            d = d.shuffle(buffer_size=config.batch_size)
        else:
            d = tf.data.TFRecordDataset(file_list)

        d = d.map(
            lambda record: tfrec_util.deserialize(record),
            num_parallel_calls=config.thread_num
        )
        d = d.batch(config.batch_size)
        return d

    return input_fn

def build_input_receiver_fn(config):
    tfrec_util = TFRecordUtil(config.height, config.width, config.in_channel, config.out_channel)
    def input_receiver_fn():
        input_example = tf.placeholder(dtype=tf.string, shape=[None], name="input_example_tensor")
        features = tfrec_util.deserialize_batch(input_example)
        receiver_tensors = {'examples': input_example}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return input_receiver_fn

def main(_):
    check_flags()

    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    config = ModelConfig(FLAGS.config_file)
    copy_config(FLAGS.config_file, FLAGS.model_dir)

    model_fn = build_model_fn(config)

    run_config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        ),
        save_summary_steps=True,
        keep_checkpoint_max=config.max_keep_ckpts,
        save_checkpoints_steps=config.save_ckpts_steps
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config
    )

    if FLAGS.do_train:
        input_fn = build_input_fn(FLAGS.train_inputs, config, is_train=True)
        estimator.train(
            input_fn=input_fn,
            max_steps=config.max_steps
        )
        # estimator.export_saved_model(
        #     os.path.join(FLAGS.model_dir, "saved_model"),
        #     serving_input_receiver_fn=build_input_receiver_fn(config)
        # )

    if FLAGS.do_eval:
        input_fn = build_input_fn(FLAGS.eval_inputs, config, is_train=False)
        estimator.evaluate(
            input_fn=input_fn
        )

    if FLAGS.do_predict:
        input_fn = build_input_fn(FLAGS.predict_inputs, config, is_train=False)    
        result = estimator.predict(
            input_fn=input_fn
        )
        if not os.path.exists(FLAGS.output_dir):
            os.mkdir(FLAGS.output_dir)
        for (_, prediction) in enumerate(result):
            example_id = prediction["example_id"][0]
            category_score = prediction["category_score"]
            instance = prediction["instance"]
            confidence_score = prediction["confidence_score"]
            class_score = prediction["class_score"]
            heading = prediction["heading"]
            height = prediction["height"]
            mask = prediction["mask"]
            
            output_predict_file = os.path.join(FLAGS.output_dir, "{:d}.pred".format(example_id))
            with tf.gfile.GFile(output_predict_file, "w") as writer:
                for r in range(config.height):
                    for c in range(config.width):
                        if mask[r][c][0] == 1:
                            writer.write("{:d} ".format(r))
                            writer.write("{:d} ".format(c))
                            writer.write("{:.3f} ".format(instance[r][c][0]))
                            writer.write("{:.3f} ".format(instance[r][c][1]))
                            writer.write("{:.3f} ".format(category_score[r][c][0]))
                            writer.write("{:.3f} ".format(confidence_score[r][c][0]))
                            writer.write("{:.3f} ".format(class_score[r][c][0]))
                            writer.write("{:.3f} ".format(class_score[r][c][1]))
                            writer.write("{:.3f} ".format(class_score[r][c][2]))
                            writer.write("{:.3f} ".format(class_score[r][c][3]))
                            writer.write("{:.3f} ".format(class_score[r][c][4]))
                            writer.write("{:.3f} ".format(heading[r][c][0]))
                            writer.write("{:.3f} ".format(heading[r][c][1]))
                            writer.write("{:.3f}\n".format(height[r][c][0]))

if __name__ == "__main__":
    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string("train_inputs", None, "Input training samples file in TFReord format")
    flags.DEFINE_string("eval_inputs", None, "Input evaluating samples file in TFReord format")
    flags.DEFINE_string("predict_inputs", None, "Input prediction samples file in TFReord format")
    flags.DEFINE_string("model_dir", None, "Directory to save models")
    flags.DEFINE_string("output_dir", None, "Output directory to save evaluate and predict results")
    flags.DEFINE_string("config_file", None, "Configs in json format")
    flags.DEFINE_bool("do_train", False, "Whether to run training")
    flags.DEFINE_bool("do_eval", False, "Whether to run evaluating")
    flags.DEFINE_bool("do_predict", False, "Whether to run evaluating")

    tf.app.run()
