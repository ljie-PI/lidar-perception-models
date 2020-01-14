#include "voxel_mapping.h"

#include <cmath>

VoxelMapping::VoxelMapping(float x_res, float x_min, float x_max,
                           float y_res, float y_min, float y_max,
                           float z_res, float z_min, float z_max)
                           : x_res_(x_res), x_min_(x_min), x_max_(x_max),
                             y_res_(y_res), y_min_(y_min), y_max_(y_max),
                             z_res_(z_res), z_min_(z_min), z_max_(z_max) {
  x_size_ = ceil((x_max_ - x_min_) / x_res_);
  y_size_ = ceil((y_max_ - y_min_) / y_res_);
  z_size_ = ceil((z_max_ - z_min_) / z_res_);
  size_ = x_size_ * y_size_ * z_size_;
}

int VoxelMapping::VoxelIndexToOffset(int x_idx, int y_idx, int z_idx) {
  if (x_idx < 0 || x_idx >= x_size_ ||
      y_idx < 0 || y_idx >= y_size_ ||
      x_idx < 0 || z_idx >= z_size_) {
        return -1;
  }
  return (x_idx * y_size_ + y_idx) * z_size_ + z_idx;
}

bool VoxelMapping::OffsetToVoxelIndex(int offset, int* x_idx, int* y_idx, int* z_idx) {
  if (offset < 0 || offset >= size_) {
    return false;
  }
  *z_idx = offset % z_size_;
  offset /= z_size_;
  *y_idx = offset % y_size_;
  offset /= y_size_;
  *x_idx = offset % x_size_;
  return true;
}

int VoxelMapping::MapToOffset(float x, float y, float z) {
  int x_idx, y_idx, z_idx;
  if (!MapToVoxelIndex(x, y, z, &x_idx, &y_idx, &z_idx)) {
    return -1;
  }
  return VoxelIndexToOffset(x_idx, y_idx, z_idx);
}

bool VoxelMapping::MapToVoxelIndex(float x, float y, float z,
                                   int* x_idx, int* y_idx, int* z_idx) {
  if (x < x_min_ || x >= x_max_ ||
      y < y_min_ || y >= y_max_ ||
      z < z_min_ || z >= z_max_) {
    return false;
  }
  *x_idx = floor((x - x_min_) / x_res_);
  *y_idx = floor((y - y_min_) / y_res_);
  *z_idx = floor((z - z_min_) / z_res_);
  return true;
}

bool VoxelMapping::VoxelCenter(int offset, float *x, float *y, float *z) {
  int x_idx, y_idx, z_idx;
  if (!OffsetToVoxelIndex(offset, &x_idx, &y_idx, &z_idx)) {
    return false;
  }
  return VoxelCenter(x_idx, y_idx, z_idx, x, y, z);
}

bool VoxelMapping::VoxelCenter(int x_idx, int y_idx, int z_idx, float *x, float *y, float *z) {
  if (x_idx < 0 || x_idx >= x_size_ ||
      y_idx < 0 || y_idx >= y_size_ ||
      z_idx < 0 || z_idx >= z_size_) {
    return false;
  }
  *x = x_min_ + (static_cast<float>(x_idx) + 0.5f) * x_res_;
  *y = y_min_ + (static_cast<float>(y_idx) + 0.5f) * y_res_;
  *z = z_min_ + (static_cast<float>(z_idx) + 0.5f) * z_res_;
  return true;
}
