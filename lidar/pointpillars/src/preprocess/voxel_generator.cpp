#include "voxel_generator.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>

#include "common/file_util.h"
#include "voxel_mapping.h"

VoxelGenerator::VoxelGenerator(const pointpillars::PointPillarsConfig& config) {
  float x_res = config.voxel_config().x_resolution();
  float y_res = config.voxel_config().y_resolution();
  float z_res = config.voxel_config().z_resolution();
  float x_range_min = config.voxel_config().x_range_min();
  float x_range_max = config.voxel_config().x_range_max();
  float y_range_min = config.voxel_config().y_range_min();
  float y_range_max = config.voxel_config().y_range_max();
  float z_range_min = config.voxel_config().z_range_min();
  float z_range_max = config.voxel_config().z_range_max();
  voxel_mapping_ = std::make_shared<VoxelMapping>(
      x_res, x_range_min, x_range_max,
      y_res, y_range_min, y_range_max,
      z_res, z_range_min, z_range_max);

  num_voxels_ = config.voxel_config().num_voxels();
  num_points_per_voxels_ = config.voxel_config().num_points_per_voxels();
  save_points_ = config.voxel_config().save_points();
  log_voxel_num_ = config.voxel_config().log_voxel_num();
  if (log_voxel_num_) {
    FileUtil::EnsureDirectory("log");
    voxel_num_ofs_.open("log/voxel_num");
    if (!voxel_num_ofs_.is_open()) {
      std::cerr << "Failed to open log/voxel_num for write" << std::endl;
      log_voxel_num_ = false;
    }
  }
  log_point_num_ = config.voxel_config().log_point_num();
  if (log_point_num_) {
    FileUtil::EnsureDirectory("log");
    voxel_num_ofs_.open("log/point_num");
    if (!point_num_ofs_.is_open()) {
      std::cerr << "Failed to open log/point_num for write" << std::endl;
      log_point_num_ = false;
    }
  }

  use_reflection_ = config.model_config().use_reflection();

  rand_shuffle_ = std::make_shared<RandomShuffle>();
}

VoxelGenerator::~VoxelGenerator() {
}

/**
 * According to the paper:
The set of pillars will be mostly empty due to sparsity
of the point cloud, and the non-empty pillars will in general
have few points in them. For example, at 0.162 m2 bins
the point cloud from an HDL-64E Velodyne lidar has 6k-9k
non-empty pillars in the range typically used in KITTI for
∼ 97% sparsity. This sparsity is exploited by imposing a
limit both on the number of non-empty pillars per sample
(P) and on the number of points per pillar (N) to create a
dense tensor of size (D, P, N). If a sample or pillar holds
too much data to fit in this tensor, the data is randomly samapled.
Conversely, if a sample or pillar has too little data to
populate the tensor, zero padding is applied.
 *
 * This function will fill voxel and voxel_coord of pointpillars::Examples
 * voxel: only contains non-empty voxels which contains LiDAR points
 * voxel_coord: (x_offset, y_offset, z_offset) of each non-empty voxel in voxel feature map
 */
bool VoxelGenerator::Generate(const LidarPointCloud& point_cloud, pointpillars::Example *example) {
  int voxel_num = 0;                              // non-empty voxels
  std::vector<Coordinate> voxel_coord;            // coordinate of each voxel
  voxel_coord.reserve(num_voxels_);
  std::vector<std::vector<size_t>> voxel_points;  // points in each voxel
  voxel_points.reserve(num_voxels_);
  std::unordered_map<int, int> coord_to_voxel_idx;// coord to voxel_idx mapping
  std::vector<int> voxel_idxs;                    // voxel indices
  voxel_idxs.reserve(num_voxels_);

  size_t pc_size = point_cloud.size();
  for (size_t pt_id = 0; pt_id < pc_size; ++pt_id) {
    const LidarPoint& point = point_cloud.at(pt_id);
    int x_offset, y_offset, z_offset;
    if (voxel_mapping_->MapToVoxelIndex(
        point.x, point.y, point.z, &x_offset, &y_offset, &z_offset)) {
      // point out of range, will ignore
      continue;
    }
    int offset = voxel_mapping_->VoxelIndexToOffset(x_offset, y_offset, z_offset);
    auto iter = coord_to_voxel_idx.find(offset);
    if (iter != coord_to_voxel_idx.end()) {
      int voxel_idx = iter->second;
      voxel_points[voxel_idx].push_back(pt_id);
    } else {
      Coordinate coord{x_offset, y_offset, z_offset};
      voxel_coord.push_back(coord);
      std::vector<size_t> new_voxel_points;
      voxel_points.push_back(new_voxel_points);
      voxel_points[voxel_num].reserve(num_points_per_voxels_);
      voxel_points[voxel_num].push_back(pt_id);
      coord_to_voxel_idx.insert(std::make_pair(offset, voxel_num));
      voxel_idxs.push_back(voxel_num++);
    }

  }

  if (voxel_num > num_voxels_) {
    // number of non-empty voxels less then num_voxels_,
    // will sample voxels
    rand_shuffle_->Shuffle(voxel_idxs.begin(), voxel_idxs.end());
    voxel_idxs.resize(num_voxels_);
  }

  for (int voffset = 0; voffset < voxel_idxs.size(); ++voffset) {
    int voxel_idx = voxel_idxs[voffset];
    Coordinate& coord = voxel_coord[voxel_idx];
    example->mutable_voxel_coord()->add_data(coord.x);
    example->mutable_voxel_coord()->add_data(coord.y);
    example->mutable_voxel_coord()->add_data(coord.z);
    float center_x, center_y, center_z;
    voxel_mapping_->VoxelCenter(coord.x, coord.y, coord.z, &center_x, &center_y, &center_z);

    auto& point_idxs = voxel_points[voxel_idx];
    int point_size = point_idxs.size();
    if (point_size > num_points_per_voxels_) {
      // more points than num_points_per_voxels_, will sample points
      rand_shuffle_->Shuffle(point_idxs.begin(), point_idxs.end());
      point_idxs.resize(num_points_per_voxels_);
    }

    for (size_t poffset = 0; poffset < point_idxs.size(); ++poffset) {
      size_t pt_idx = point_idxs[poffset];
      const LidarPoint& point = point_cloud.at(pt_idx);
      float distance = sqrt(point.x * point.x + point.y * point.y);
      example->mutable_voxel()->add_data(point.x);
      example->mutable_voxel()->add_data(point.y);
      example->mutable_voxel()->add_data(point.z);
      if (use_reflection_) {
        example->mutable_voxel()->add_data(point.intensity);
      }
      example->mutable_voxel()->add_data(distance);
      // [x, y, z] to pillar center
      example->mutable_voxel()->add_data(point.x - center_x);
      example->mutable_voxel()->add_data(point.y - center_y);
      example->mutable_voxel()->add_data(point.z - center_z);

      if (save_points_) {
        auto exam_point = example->add_point();
        exam_point->set_x(point.x);
        exam_point->set_y(point.y);
        exam_point->set_z(point.z);
        exam_point->set_reflection(point.intensity);
        exam_point->set_distance(distance);
      }
    }

    example->mutable_voxel_points()->add_data(point_size);

    if (point_size < num_points_per_voxels_) {
      // less points than num_points_per_voxels_, will padding with zeros
      for (size_t poffset = point_size; poffset < num_points_per_voxels_; ++poffset) {
        for (int dim = 0; dim < POINT_DIM; ++dim) {
          example->mutable_voxel()->add_data(0.0);
        }
      }
    }
  }

  example->mutable_voxel_coord()->add_shape(voxel_num);
  example->mutable_voxel_coord()->add_shape(COORD_DIM);
  example->mutable_voxel()->add_shape(voxel_num);
  example->mutable_voxel()->add_shape(num_points_per_voxels_);
  example->mutable_voxel()->add_shape(POINT_DIM);
  example->mutable_voxel_points()->add_shape(voxel_num);
}