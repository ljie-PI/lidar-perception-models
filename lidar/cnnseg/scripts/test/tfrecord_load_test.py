#!/usr/bin/env python

import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

def print_usage():
    print("""
    Usage: python tfrecord_load_test.py <input> <output>
    """)

def load_and_save(in_file, out_file):
    record_iter = tf.python_io.tf_record_iterator(path=in_file)
    with open(out_file, "w") as fout:
        for record in record_iter:
            example = tf.train.Example.FromString(record)
            example_json = MessageToJson(example)
            fout.write(example_json)
            fout.flush()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(-1)
    load_and_save(sys.argv[1], sys.argv[2])