#include "preprocessor.h"

#include <iostream>
#include <pcl/io/pcd_io.h>

#include "common/pcl_defs.h"
#include "common/protobuf_util.h"

PreProcessor::PreProcessor(pointpillars::PointPillarsConfig& config)
    : config_(config) {
  voxel_generator_ = std::make_shared<VoxelGenerator>(config_);
  target_assigner_ = std::make_shared<TargetAssigner>(config_);
}

bool PreProcessor::SaveExample(const pointpillars::Example& example,
                               const std::string& example_id,
                               const std::string& output_dir) {
  std::string output_file = output_dir + "/" + example_id;
  return ProtobufUtil::SaveToBinaryFile(example, output_file);
}

bool PreProcessor::Process(const std::string& example_id, const std::string& pcd_file,
                           const std::string& label_file, const std::string& output_dir) {

  LidarPointCloud point_cloud;
  if (pcl::io::loadPCDFile(pcd_file, point_cloud) < 0) {
    std::cerr << "Failed to load pcd file: " << pcd_file << std::endl;
    return false;
  }
  Label label;
  if (!label.FromFile(label_file)) {
    std::cerr << "Failed to load label file: " << label_file << std::endl;
    return false;
  }

  pointpillars::Example example;
  if (!voxel_generator_->Generate(point_cloud, &example)) {
    std::cerr << "Failed to generate voxels for example: " << example_id << std::endl;
    return false;
  }

  if (!target_assigner_->Assign(point_cloud, label, &example)) {
    std::cerr << "Failed to generate labels for example: " << example_id << std::endl;
    return false;
  }

  if (!SaveExample(example, example_id, output_dir)) {
    std::cerr << "Failed to save example: " << example_id << std::endl;
    return false;
  }

  return true;
}
