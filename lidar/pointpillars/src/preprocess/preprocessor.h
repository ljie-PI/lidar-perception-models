#pragma once

#include "voxel_generator.h"
#include "target_assigner.h"
#include "pp_config.pb.h"

class PreProcessor {
 public:
  explicit PreProcessor(pointpillars::PointPillarsConfig& config);
  ~PreProcessor() = default;

  bool Process(const std::string& example_id, const std::string& pcd_file,
               const std::string& label_file, const std::string& output_dir);

 private:
  bool SaveExample(const pointpillars::Example& example,
                   const std::string& example_id,
                   const std::string& output_dir);

  pointpillars::PointPillarsConfig config_;
  std::shared_ptr<VoxelGenerator> voxel_generator_;
  std::shared_ptr<TargetAssigner> target_assigner_;
};