#!/bin/bash

dir=`dirname $0`

for test_file in `ls $dir/*_test.py`; do
    echo "run test: $test_file"
    python $test_file
done