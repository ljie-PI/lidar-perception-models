#!/usr/bin/env python

import os
import argparse
import shutil


def split(pcd_dir, label_dir, parts):
    pcd_part_dirs = []
    label_part_dirs = []
    for i in range(parts):
        pcd_part_dir = "{}_part{}".format(pcd_dir, i)
        if not os.path.exists(pcd_part_dir):
            os.makedirs(pcd_part_dir)
        pcd_part_dirs.append(pcd_part_dir)
        label_part_dir = "{}_part{}".format(label_dir, i)
        if not os.path.exists(label_part_dir):
            os.makedirs(label_part_dir)
        label_part_dirs.append(label_part_dir)
    file_num = 0
    for pcd_file in os.listdir(pcd_dir):
        example_id = pcd_file.split(".")[0]
        part = file_num % parts
        src_pcd_file = "{}/{}".format(pcd_dir, pcd_file)
        dst_pcd_file = "{}/{}.pcd".format(pcd_part_dirs[part], example_id)
        os.system("cp {} {}".format(src_pcd_file, dst_pcd_file))
        src_label_file = "{}/{}.label".format(label_dir, example_id)
        dst_label_file = "{}/{}.label".format(label_part_dirs[part], example_id)
        os.system("cp {} {}".format(src_label_file, dst_label_file))
        file_num += 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--parts", type=int, default=4, required=True)
    arg_parser.add_argument("--pcd_dir", required=True)
    arg_parser.add_argument("--label_dir", required=True)
    args = arg_parser.parse_args()

    split(args.pcd_dir.rstrip("/"), args.label_dir.rstrip("/"), args.parts)
    