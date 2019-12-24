#!/usr/bin/env python

import sys
import tensorflow as tf

from multiprocessing import Process, Queue
from time import sleep
from glob import glob
from tfrecord_util import TFRecordUtil

def check_flags():
    if FLAGS.feature_dir is None:
        raise ValueError("No feature directory specified")
    if FLAGS.output_dir is None:
        raise ValueError("No output directory specified")
    if FLAGS.height is None:
        raise ValueError("Height is not specified")
    if FLAGS.width is None:
        raise ValueError("Width is not specified")
    if FLAGS.in_channel is None:
        raise ValueError("Input channel is not specified")
    if FLAGS.record_tag is None:
        raise ValueError("record_tag is not specified")
    if FLAGS.out_channel is None:
        raise ValueError("Output channel is not specified")
    if FLAGS.output_parts < FLAGS.worker_num:
        raise ValueError("Output parts should be larger than worker number")

class SparseMultiChanData(object):
    def __init__(self, indices, values, shapes):
        self._nonempty_indices = []
        self._nonempty_values = []
        idx_val_zip = list(zip(indices, values))

        # sparse tensor requires sorted indices, so sort by indices first
        sorted_idx_val = sorted(idx_val_zip, key=lambda x: x[0][0] * 10000 + x[0][1])

        for srtd_idx, chan_values in sorted_idx_val:
            for c in range(shapes[2]):
                self._nonempty_indices.append((srtd_idx[0], srtd_idx[1], c))
                self._nonempty_values.append(chan_values[c])
        self._shapes = shapes

    def nonempyt_height_indices(self):
        return [idx[0] for idx in self._nonempty_indices]

    def nonempty_width_indices(self):
        return [idx[1] for idx in self._nonempty_indices]

    def nonempty_chan_indices(self):
        return [idx[2] for idx in self._nonempty_indices]

    def nonempty_values(self):
        return self._nonempty_values

    def shapes(self):
        return self._shapes

class Example(object):
    def __init__(self, features, labels, sample_id,
                 height, width, feat_channel, label_channel):
        self._features = features
        self._labels = labels
        self._sample_id = sample_id
        self._tfrecord_util = TFRecordUtil(height, width, feat_channel, label_channel)

    def to_tfrecord(self):
        return self._tfrecord_util.serialize(
            self._features.nonempyt_height_indices(),
            self._features.nonempty_width_indices(),
            self._features.nonempty_chan_indices(),
            self._features.nonempty_values(),
            self._labels.nonempyt_height_indices(),
            self._labels.nonempty_width_indices(),
            self._labels.nonempty_chan_indices(),
            self._labels.nonempty_values(),
            self._sample_id
        )

class DataSet(object):
    def __init__(self, feature_dir, label_dir, height, width, in_channel, out_channel, record_tag):
        self._feature_dir = feature_dir
        self._label_dir = label_dir
        self._height = height
        self._width = width
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._record_tag = record_tag

    def convert_to_tfrecords(self, tfrecord_dir):
        writers = []
        for i in range(FLAGS.output_parts):
            writer = tf.io.TFRecordWriter("{}/part-{}.tfrecord".format(tfrecord_dir, i))
            writers.append(writer)

        example_ids = self._get_input_examples_ids()
        wtr_idx = 0
        for example_id in example_ids:
            example = self.parse_example(example_id)
            writers[wtr_idx].write(example.to_tfrecord())
            if wtr_idx == FLAGS.output_parts - 1:
                wtr_idx = 0
            else:
                wtr_idx += 1
        for writer in writers:
            writer.flush()
            writer.close()

    def convert_to_tfrecords_parallel(self, tfrecord_dir):
        msg_queue = Queue(1000)
        processes = []
        for i in range(FLAGS.worker_num):
            file_indices = list(filter(
                lambda idx: idx % FLAGS.worker_num == i,
                range(FLAGS.output_parts)
            ))
            print("process " + str(i) + " will write to " + str(file_indices))
            process = Process(
                target=lambda q: self._convert_process(q, tfrecord_dir, file_indices),
                args=(msg_queue,)
            )
            processes.append(process)
            process.start()

        example_ids = self._get_input_examples_ids()
        for example_id in example_ids:
            if msg_queue.full():
                print("queue is full, sleep for 20 seconds")
                sleep(20)
                continue
            msg_queue.put(example_id)

        # send stop signal to processes
        for i in range(len(processes)):
            msg_queue.put(None)

        for process in processes:
            process.join()

    def parse_example(self, example_id):
        features = self._parse_features("{}/{}.txt".format(self._feature_dir, example_id))
        if self._label_dir is None:
            labels = SparseMultiChanData([], [], shapes=(0, 0, 0))
        else:
            labels = self._parse_labels("{}/{}.txt".format(self._label_dir, example_id))
        example = Example(features, labels, int(example_id),
                          self._height, self._width, self._in_channel, self._out_channel)
        return example

    def _convert_process(self, msg_queue, tfrecord_dir, file_indices):
        flen = len(file_indices)
        writers = []
        for i in range(flen):
            writer = tf.io.TFRecordWriter("{}/{}-part-{}.tfrecord".format(tfrecord_dir, self._record_tag, file_indices[i]))
            writers.append(writer)
        wtr_idx = 0
        while True:
            if msg_queue.empty():
                print("queue is empty, sleep for 1 second")
                sleep(1)
                continue
            example_id = msg_queue.get()
            if example_id is None:
                for writer in writers:
                    writer.flush()
                    writer.close()
                return
            try:
                example = self.parse_example(example_id)
            except Exception as e:
                print("ERROR: Failed to parse example {}".format(example_id))
                continue
            writers[wtr_idx].write(example.to_tfrecord())
            if wtr_idx == flen - 1:
                wtr_idx = 0
            else:
                wtr_idx += 1

    def _get_input_examples_ids(self):
        example_ids = []
        feat_files = glob("{}/*.txt".format(self._feature_dir))
        for feat_file in feat_files:
            example_id = self._parse_example_id(feat_file)
            example_ids.append(example_id)
        return example_ids

    def _parse_features(self, feat_file):
        with open(feat_file) as ffeat:
            indices = []
            values = []
            for _, line in enumerate(ffeat):
                parts = line.strip().split(" ")
                row_idx = int(parts[0])
                col_idx = int(parts[1])
                if row_idx >= self._height or col_idx >= self._width:
                    continue
                indices.append([row_idx, col_idx])
                max_height = float(parts[2])
                mean_height = float(parts[3]) 
                count_data = float(parts[4])
                direction_data = float(parts[5])
                top_intensity_data = float(parts[6])
                mean_intensity_data = float(parts[7])
                distance_data = float(parts[8])
                nonempty_data = float(parts[9])
                value = [max_height, mean_height, count_data]
                if FLAGS.use_constant:
                    value.append(direction_data)
                if FLAGS.use_intensity:
                    value.extend([top_intensity_data, mean_intensity_data])
                if FLAGS.use_constant:
                    value.append(distance_data)
                value.append(nonempty_data)
                values.append(value)
        return SparseMultiChanData(indices, values,
                                   shapes=(self._height, self._width, self._in_channel))

    def _parse_example_id(self, feat_file):
        parts = feat_file.split("/")
        file_name = parts[-1]
        example_id = file_name[0:file_name.find(".txt")]
        return example_id

    def _score_to_label(self, score, thr=0.5):
        if score >= thr:
            return 1
        return 0

    def _parse_labels(self, label_file):
        with open(label_file) as flbl:
            indices = []
            values = []
            for _, line in enumerate(flbl):
                parts = line.strip().split(" ")
                row_idx = int(parts[0])
                col_idx = int(parts[1])
                if row_idx >= self._height or col_idx >= self._width:
                    continue
                indices.append([row_idx, col_idx])
                instance_0 = float(parts[2])
                instance_1 = float(parts[3])
                category_score = float(parts[4])
                confidence_score = float(parts[5])
                classify_score_0 = float(parts[6])
                classify_score_1 = float(parts[7])
                classify_score_2 = float(parts[8])
                classify_score_3 = float(parts[9])
                classify_score_4 = float(parts[10])
                heading_0 = float(parts[11])
                heading_1 = float(parts[12])
                height = float(parts[13])
                value = (category_score, instance_0, instance_1, confidence_score,
                              classify_score_0, classify_score_1, classify_score_2,
                              classify_score_3, classify_score_4,
                              heading_0, heading_1, height)
                values.append(value)
        return SparseMultiChanData(indices, values,
                                   shapes=(self._height, self._width, self._out_channel))

def main(_):
    check_flags()
    dataset = DataSet(
        FLAGS.feature_dir,
        FLAGS.label_dir,
        FLAGS.height,
        FLAGS.width,
        FLAGS.in_channel,
        FLAGS.out_channel,
        FLAGS.record_tag
    )
    # dataset.convert_to_tfrecords(FLAGS.output_dir)
    dataset.convert_to_tfrecords_parallel(FLAGS.output_dir)

if __name__ == "__main__":
    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string("feature_dir", None, "Input directory containing example features")
    flags.DEFINE_string("label_dir", None, "Input directory containinig example labels")
    flags.DEFINE_string("output_dir", None, "Output directory of TFRecored")
    flags.DEFINE_string("record_tag", "baidu", "Tf record tag for different data source")
    flags.DEFINE_integer("output_parts", None, "Output parts")
    flags.DEFINE_integer("height", 480, "Height of example")
    flags.DEFINE_integer("width", 480, "Width of example")
    flags.DEFINE_integer("in_channel", 6, "Number of input feature channels")
    flags.DEFINE_integer("out_channel", 12, "Number of output label channels")
    flags.DEFINE_integer("worker_num", 10, "Number of work processor")
    flags.DEFINE_boolean("use_intensity", False, "Whether to use intensity feature")
    flags.DEFINE_boolean("use_constant", True, "Whether to use constant feature")
    tf.app.run()
