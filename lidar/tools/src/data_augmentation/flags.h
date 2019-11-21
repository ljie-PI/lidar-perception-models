#pragma once

#include <cmath>

#include "gflags/gflags.h"


static bool ValidateStringNotEmpty(const char *flagname, const std::string &value) {
  return !value.empty();
}

static double default_trans_ratio = 0.1;

DEFINE_string(input_pcd_dir, "", "directory of input origin pcd files");
DEFINE_validator(input_pcd_dir, &ValidateStringNotEmpty);
DEFINE_string(input_label_dir, "", "directory of input origin label files");
DEFINE_validator(input_label_dir, &ValidateStringNotEmpty);

DEFINE_string(output_pcd_dir, "", "directory of output augmented pcd files");
DEFINE_validator(output_pcd_dir, &ValidateStringNotEmpty);
DEFINE_string(output_label_dir, "", "directory of output augmented label files");
DEFINE_validator(output_label_dir, &ValidateStringNotEmpty);

DEFINE_bool(reflect_x, false,
            "augment input point cloud data by reflecting it over x axis");
DEFINE_double(reflect_x_ratio, default_trans_ratio,
              "ratio of data augmented by reflecting over x axis");

DEFINE_bool(reflect_y, false,
            "augment input point cloud data by reflecting it over y axis");
DEFINE_double(reflect_y_ratio, default_trans_ratio,
              "ratio of data augmented by reflecting over y axis");

DEFINE_bool(rotate_z, false,
            "augment input point cloud data by rotating around z axis");
DEFINE_double(rotate_z_ratio, default_trans_ratio,
              "ratio of data augmented by rotating around z axis");
DEFINE_double(min_rot_angle, -M_PI / 10.0, "minimum rotate angle");
DEFINE_double(max_rot_angle, M_PI / 10.0, "maximum rotate angle");

DEFINE_bool(scale, false,
            "augment input point cloud data by scaling it");
DEFINE_double(scale_ratio, default_trans_ratio,
              "ratio of data augmented by scaling it");
DEFINE_double(min_scale, 0.95, "minimum scale factor");
DEFINE_double(max_scale, 1.05, "maximum scale factor");

DEFINE_bool(move, false,
            "augment input point cloud data by moving whole point cloud");
DEFINE_double(move_ratio, default_trans_ratio,
              "ratio of data augmented by moving whole point cloud");
DEFINE_double(move_mean, 0, "mean move distance");
DEFINE_double(move_std, 0.25, "standard divation of move distance");

DEFINE_bool(down_sample, false,
            "augment input point cloud data by down sampling");
DEFINE_double(down_sample_ratio, default_trans_ratio,
              "ratio of data augmented by down sampling");
DEFINE_double(down_sample_range, 10, "range in which obstacles are applied down sampling");
DEFINE_double(down_sample_factor, 0.6, "down sample factor");

DEFINE_bool(up_sample, false,
            "augment input point cloud data by up sampling");
DEFINE_double(up_sample_ratio, default_trans_ratio,
              "ratio of data augmented by up sampling");
DEFINE_double(up_sample_range, 10, "range in which obstacles are applied up sampling");
DEFINE_double(up_sample_factor, 2.0, "up sample factor");

DEFINE_bool(ground_filter, false,
            "augment input point cloud data by removing points on ground");
DEFINE_double(ground_filter_ratio, default_trans_ratio,
              "ratio of data augmented by ground filter");
DEFINE_double(ground_height, -0.2, "ground height below which points will be ignored");

DEFINE_bool(rotate_obs, false,
            "augment input point cloud data by rotating obstacles");
DEFINE_double(rotate_obs_ratio, default_trans_ratio,
              "ratio of data augmented by rotating obstacles");
DEFINE_double(min_obs_rot_angle, -M_PI / 10.0, "minimum rotate angle of obstacles");
DEFINE_double(max_obs_rot_angle, M_PI / 10.0, "maximum rotate angle of obstacles");

DEFINE_bool(scale_obs, false,
            "augment input point cloud data by scaling obstacles");
DEFINE_double(scale_obs_ratio, default_trans_ratio,
              "ratio of data augmented by scaling obstacles");
DEFINE_double(min_obs_scale, 0.95, "minimum scale factor for obstacles");
DEFINE_double(max_obs_scale, 1.05, "maximum scale factor for obstacles");

DEFINE_bool(move_obs, false,
            "augment input point cloud data by moving obstacles");
DEFINE_double(move_obs_ratio, default_trans_ratio,
              "ratio of data augmented by adding noise");
DEFINE_double(move_obs_mean, 0, "mean of move distance");
DEFINE_double(move_obs_std, 0.25, "standard deviation of move distance");
