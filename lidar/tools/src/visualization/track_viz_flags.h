#pragma once

#include "gflags/gflags.h"

static bool ValidateStringNotEmpty(const char *flagname, const std::string &value) {
  return !value.empty();
}

DEFINE_string(input_pcd_dir, "", "directory of input origin pcd files");
DEFINE_validator(input_pcd_dir, &ValidateStringNotEmpty);
DEFINE_string(input_pose_dir, "", "directory of input pose files");
DEFINE_validator(input_pose_dir, &ValidateStringNotEmpty);
DEFINE_string(input_track_dir, "", "directory of input track result files");
DEFINE_validator(input_track_dir, &ValidateStringNotEmpty);
DEFINE_string(id_list, "", "id lists of examples to be visualized");
DEFINE_validator(id_list, &ValidateStringNotEmpty);
DEFINE_double(spin_times, 0,
    "display spin_times of each frame, by default(0) will always display unless close the window");