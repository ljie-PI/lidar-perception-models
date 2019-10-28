#!/usr/bin/env python

import os
import argparse
import shutil

ID_OFFSET = {
    "reflect_x": 10000000,
    "reflect_y": 20000000,
    "rotate_z": 30000000,
    "scale": 40000000,
    "move": 50000000,
    "ground_filter": 60000000,
    "down_sample": 70000000,
    "up_sample": 80000000
}


def merge_train_data(raw_pcd_dir, raw_label_dir,
                     aug_pcd_dir, aug_label_dir,
                     merge_pcd_dir, merge_label_dir):
    if not os.path.exists(raw_pcd_dir) or not os.path.exists(raw_label_dir) \
            or not os.path.exists(aug_pcd_dir) or not os.path.exists(aug_label_dir):
        return
    if not os.path.exists(merge_pcd_dir):
        os.makedirs(merge_pcd_dir)
    if not os.path.exists(merge_label_dir):
        os.makedirs(merge_label_dir)

    for pcd_file in os.listdir(raw_pcd_dir):
        if pcd_file[-4:] == ".pcd":
            src_file = os.path.join(raw_pcd_dir, pcd_file)
            dst_file = os.path.join(merge_pcd_dir, pcd_file)
            shutil.copyfile(src_file, dst_file)
    print("Copied raw pcd files into merged pcd direcory")

    for label_file in os.listdir(raw_label_dir):
        if label_file[-6:] == ".label":
            src_file = os.path.join(raw_label_dir, label_file)
            dst_file = os.path.join(merge_label_dir, label_file)
            shutil.copyfile(src_file, dst_file)
    print("Copied raw label files into merged label direcory")

    aug_sub_pcd_dirs = os.listdir(aug_pcd_dir)
    for i in range(len(aug_sub_pcd_dirs)):
        sub_pcd_dir = aug_sub_pcd_dirs[i]
        for pcd_file in os.listdir(os.path.join(aug_pcd_dir, sub_pcd_dir)):
            if pcd_file[-4:] == ".pcd":
                eid = ID_OFFSET[sub_pcd_dir] + int(pcd_file[:-4])
                file_name = "{:d}.pcd".format(eid)
                src_file = os.path.join(aug_pcd_dir, sub_pcd_dir, pcd_file)
                dst_file = os.path.join(merge_pcd_dir, file_name)
                shutil.copyfile(src_file, dst_file)
    print("Copied augmented pcd files into merged pcd direcory")

    aug_sub_label_dirs = os.listdir(aug_label_dir)
    for sub_label_dir in aug_sub_label_dirs:
        for label_file in os.listdir(os.path.join(aug_label_dir, sub_label_dir)):
            if label_file[-6:] == ".label":
                eid = ID_OFFSET[sub_label_dir] + int(label_file[:-6])
                file_name = "{:d}.label".format(eid)
                src_file = os.path.join(aug_label_dir, sub_label_dir, label_file)
                dst_file = os.path.join(merge_label_dir, file_name)
                shutil.copyfile(src_file, dst_file)
    print("Copied augmented label files into merged label direcory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_pcd_dir', help='directory of raw pcd files', required=True)
    parser.add_argument('--raw_label_dir', help='directory of raw label files', required=True)
    parser.add_argument('--aug_pcd_dir', help='directory of augmented pcd files', required=True)
    parser.add_argument('--aug_label_dir', help='directory of augmented label files', required=False)
    parser.add_argument('--merge_pcd_dir', help='directory of merged pcd files', required=False)
    parser.add_argument('--merge_label_dir', help='directory of merged label files', required=False)
    args = parser.parse_args()
    merge_train_data(args.raw_pcd_dir, args.raw_label_dir,
                     args.aug_pcd_dir, args.aug_label_dir,
                     args.merge_pcd_dir, args.merge_label_dir)
