/*
 * used to generate test example file for python unit test
 */
#include <iostream>

#include <gflags/gflags.h>

#include "common/file_util.h"
#include "common/string_util.h"
#include "common/protobuf_util.h"
#include "preprocess/preprocessor.h"
#include "test_utils.h"

DEFINE_string(config_file, "", "config_file");
DEFINE_string(point_file, "", "point file text format");
DEFINE_string(label_file, "", "label file");
DEFINE_string(out_file, "", "output example file in protobuf format");
DEFINE_bool(output_txt, false, "whether to output text format");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  pointpillars::PointPillarsConfig pp_config;
  ProtobufUtil::ParseFromASCIIFile(FLAGS_config_file, &pp_config);

  PreProcessor preprocessor(pp_config);

  std::vector<std::string> pcd_files;

  if (!FileUtil::Exists(FLAGS_point_file)) {
    std::cerr << "Point file(" << FLAGS_point_file << ") does not exist" << std::endl;
    return -1;
  }
  if (!FileUtil::Exists(FLAGS_label_file)) {
    std::cerr << "Label file(" << FLAGS_label_file << ") does not exist" << std::endl;
    return -1;
  }
  
  LidarPointCloud point_cloud;
  if (!LoadPointCloud(FLAGS_point_file, &point_cloud)) {
    std::cerr << "Failed to load point cloud from file: " << FLAGS_point_file << std::endl;
    return -1;
  }
  Label label;
  if (!label.FromFile(FLAGS_label_file)) {
    std::cerr << "Failed to load label from file: " << FLAGS_label_file << std::endl;
    return -1;
  }
  pointpillars::Example example;
  auto voxel_generator = std::make_shared<VoxelGenerator>(pp_config);
  auto target_assigner = std::make_shared<TargetAssigner>(pp_config);
  if (!voxel_generator->Generate(point_cloud, &example)) {
    std::cerr << "Failed to generate voxels " << std::endl;
    return -1;
  }
  if (!target_assigner->Assign(point_cloud, label, &example)) {
    std::cerr << "Failed to generate labels" << std::endl;
    return -1;
  }
  if (FLAGS_output_txt) {
    return ProtobufUtil::SaveToASCIIFile(example, FLAGS_out_file);
  } else {
    return ProtobufUtil::SaveToBinaryFile(example, FLAGS_out_file);
  }
}
