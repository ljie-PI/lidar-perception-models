#pragma once

#include <fstream>

#include "common/random.h"
#include "common/pcl_defs.h"
#include "voxel_mapping.h"
#include "pp_config.pb.h"
#include "pp_example.pb.h"

struct Coordinate {
  int x;
  int y;
  int z;
};

class VoxelGenerator {
 public:
  VoxelGenerator(const pointpillars::PointPillarsConfig& config);
  ~VoxelGenerator();

  bool Generate(const LidarPointCloud& point_cloud, pointpillars::Example *example);

 private:
  std::shared_ptr<VoxelMapping> voxel_mapping_;
  int num_voxels_;
  int num_points_per_voxels_;
  bool save_points_;

  bool log_voxel_num_;
  std::ofstream voxel_num_ofs_;

  bool log_point_num_;
  std::ofstream point_num_ofs_;

  std::shared_ptr<RandomShuffle> rand_shuffle_;

  const int COORD_DIM = 3;
  const int POINT_DIM = 5;
};