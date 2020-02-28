# perception-models
Models Used in Neolix Perception Module

## Dependencies

### C++

+ Boost
+ PCL
+ Eigen
+ GFlags
+ GTests

### Python
+ Tensorflow (1.14.0, not adapted for 2.0 or later)

## Commands

### Build Tools
```
cd ${PROJECT_ROOT}/
mkdir build && cd build
cmake ..
make -j8
```

### Data Augmentation
```
cd ${PROJECT_ROOT}/build
bin/data_aug \
  --input_pcd_dir=/nfs/nas/Perception/cnnseg_train_val_29719/train/pcd \
  --input_label_dir=/nfs/nas/Perception/cnnseg_train_val_29719/train/label \
  --output_pcd_dir=/data/perception/cnnseg/data_aug/pcd \
  --output_label_dir=/data/perception/cnnseg/data_aug/label \
  --reflect_x --reflect_y --rotate_z \
  --scale --move --down_sample --up_sample --ground_filter
```
Please refer to `${PROJECT_ROOT}/lidar/tools/src/data_augmentation/flags.h` for more options.

**To merge augmentated data**

```
cd ${PROJECT_ROOT}/lidar/tools
scripts/merge_data_aug.py \
  --raw_pcd_dir=/nfs/nas/Perception/cnnseg_train_val_29719/train/pcd \
  --raw_label_dir=/nfs/nas/Perception/cnnseg_train_val_29719/train/label \
  --aug_pcd_dir=/data/perception/cnnseg/data_aug/pcd \
  --aug_label_dir=/data/perception/cnnseg/data_aug/label \
  --merge_pcd_dir=/data/perception/cnnseg/train_merged/pcd \
  --merge_label_dir=/data/perception/cnnseg/train_merged/label
```

### Calculate Metrics

### Visualization
**To visualize segmentation result**
```
cd ${PROJECT_ROOT}/build
bin/seg_view \
  --input_pcd_dir=/path/to/input_pcd_dir \
  --input_label_dir=/path/to/input_label_dir \
  --id_list=id1,id2
```
Please refer to `${PROJECT_ROOT}/lidar/tools/src/visualization/seg_viz_flags.h` for more options.

**To visualize tracking result**
```
cd ${PROJECT_ROOT}/build
bin/track_view \
  --input_pcd_dir=/path/to/input_pcd_dir \
  --input_pose_dir=/path/to/input_pose_dir \
  --input_track_dir=/path/to/input_track_dir \
  --id_list=id1,id2
```
Please refer to `${PROJECT_ROOT}/lidar/tools/src/visualization/track_viz_flags.h` for more options.

### CNN Segmentation Model

**To generate cnnseg features**
```
cd ${PROJECT_ROOT}/build
bin/cnnseg_feat_gen \
  --input_pcd_dir=/data/perception/cnnseg/train_merged/pcd \
  --output_dir=/data/perception/cnnseg/train_merged/cnnseg_feature \
  --height=480 --width=480 --range=40 \
  --min_height=-2.5 --max_height=2.5 --channel_num=8
```

**To generate cnnseg labels**
```
cd ${PROJECT_ROOT}/lidar/cnnseg
scripts/label_generate/label_generate_tool.py \
  --input_path=/data/perception/cnnseg/train_merged/label \
  --output_path=/data/perception/cnnseg/train_merged/cnnseg_label \
  --width=480 --height=480 --range=40.0
```

**Convert training data to TFRecord for efficient training IO**
```
cd ${PROJECT_ROOT}/lidar/cnnseg/scripts
./dataset.py --feature_dir=/path/to/cnnseg_feature \                                                       
  --label_dir=/path/to/cnnseg_label \
  --output_dir=./data/tfrecord_train_data \
  --height=480 --width=480 --in_channel=6 --out_channel=12 \
  --output_parts=100
```

**To train the model**
```
cd ${PROJECT_ROOT}/lidar/cnnseg/scripts
./model_run.py --do_train \                                                                                       
  --config_file=./data/config/baseline.config.json \
  --model_dir=./data/models/aug_baseline_concat3 \
  --train_inputs="./data/tfrecord_train_data/*.tfrecord" >> logs/aug_baseline_concat3.log 2>&1 &
```

**To convert model for TensorRT runtime**
```
cd ${PROJECT_ROOT}/lidar/cnnseg/scripts
./model_convert.py --config_file=./data/config/baseline.config.json --model_dir=./data/models/aug_baseline_concat3
```

### PointPillars Model

**To preprocess data for training**
```
cd ${PROJECT_ROOT}/build
bin/pp_example_gen \
  --config_file=../lidar/pointpillars/config/pp_config.pb.txt \
  --input_pcd_dir=/data/perception/pointpillars/test/pcd \
  --input_label_dir=/data/perception/pointpillars/test/labels \
  --output_dir=/data/perception/pointpillars/test/preprocess \
  --output_anchor
```

**To train the model**
```
./model_run.py \
  --action train \
  --config /nfs/data/perception/pointpillars/configs/pp_config.pb.txt \
  --train_data_path /nfs/data/perception/pointpillars/data/train \
  --eval_data_path /nfs/data/perception/pointpillars/data/eval \
  --model_path /nfs/data/perception/pointpillars/models/baseline
```

**To predict with model**
```
./model_run.py \
  --action predict \
  --config /volume1/data/pointpillars/configs/pp_config_lr_0.0005.pb.txt \
  --pred_data_path /volume1/data/pointpillars/test/preprocess \
  --pred_output /volume1/data/pointpillars/test/preds/baseline_lr_0.0005 \
  --model_path /volume1/data/pointpillars/models/baseline_lr_0.0005
```