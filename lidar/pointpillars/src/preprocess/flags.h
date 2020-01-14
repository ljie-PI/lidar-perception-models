#include <iostream>
#include <gflags/gflags.h>

static bool ValidateStringNotEmpty(const char *flagname, const std::string &value) {
  if (value.empty()) {
    std::cerr << flagname << " is not specified!" << std::endl;
    return false;
  }
  return true;
}

DEFINE_string(input_pcd_dir, "", "directory of input origin pcd files");
DEFINE_validator(input_pcd_dir, &ValidateStringNotEmpty);
DEFINE_string(input_label_dir, "", "directory of input origin label files");
DEFINE_validator(input_label_dir, &ValidateStringNotEmpty);
DEFINE_string(config_file, "", "config file for PointPillars model");
DEFINE_validator(config_file, &ValidateStringNotEmpty);
DEFINE_string(output_dir, "", "directory of output processed examples");
DEFINE_validator(output_dir, &ValidateStringNotEmpty);
DEFINE_bool(output_anchor, false, "whether to output anchor results");
