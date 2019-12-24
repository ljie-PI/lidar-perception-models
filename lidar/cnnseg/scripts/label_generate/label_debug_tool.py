import numpy as np
import cv2
import sys
import os
import argparse


def show_feature_label(example_feature_path, example_label_path, example_id, featrue_size):
    feature_point_color = (0, 0, 255)  # red
    label_point_color = (0, 255, 0)  # green
    point_size = 1
    thickness = 1
    img = np.zeros((int(featrue_size), int(featrue_size), 3), np.uint8)
    img.fill(255)
    confidence_thr = 0.1
    category_thr = 0.0
    example_id = example_id+".txt"
    feature_points_list = []
    label_points_list = []

    with open(os.path.join(example_feature_path, example_id)) as f_in:
        for line in f_in:
            strs = line.strip().split(' ')
            feature_points_list.append((int(strs[0]), int(strs[1])))
    with open(os.path.join(example_label_path, example_id)) as l_in:
        for line in l_in:
            strs = line.strip().split(' ')
            category_score = float(strs[4])
            confidence_score = float(strs[5])
            # if confidence_score > confidence_thr and category_score > category_thr:
            label_points_list.append((int(strs[0]), int(strs[1])))
    for point in feature_points_list:
        cv2.circle(img, point, point_size, feature_point_color, thickness)
    for point in label_points_list:
        cv2.circle(img, point, point_size, label_point_color, thickness)

    f_in.close()
    l_in.close()

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for label debug')
    parser.add_argument('--feature_path', '-f',
                        help='input path of feature file', required=True)
    parser.add_argument('--label_path',   '-l',
                        help='output path of label file', required=True)
    parser.add_argument('--example_id',   '-id',
                        help='file id to debug', required=True)
    parser.add_argument('--featrue_size', '-fs',
                        help='feature size for showing', required=False, default=480)

    args = parser.parse_args()
    show_feature_label(args.feature_path, args.label_path,
                       args.example_id, args.featrue_size)
