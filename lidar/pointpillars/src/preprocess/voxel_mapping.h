#pragma once

class VoxelMapping {
 public:
  VoxelMapping(float x_res, float x_min, float x_max,
               float y_res, float y_min, float y_max,
               float z_res, float z_min, float z_max);
  ~VoxelMapping() = default;


  int VoxelIndexToOffset(int x_idx, int y_idx, int z_idx);
  bool OffsetToVoxelIndex(int offset, int* x_idx, int* y_idx, int* z_idx);

  int MapToOffset(float x, float y, float z);
  bool MapToVoxelIndex(float x, float y, float z,
                       int* x_idx, int* y_idx, int* z_idx);

  bool VoxelCenter(int offset, float* x, float* y, float* z);
  bool VoxelCenter(int x_idx, int y_idx, int z_idx,
                   float* x, float* y, float* z);

  int Size() { return size_; }
  int XSize() { return x_size_; }
  int YSize() { return y_size_; }
  int ZSize() { return z_size_; }

  float XMin() { return x_min_; }
  float XMax() { return x_max_; }
  float YMin() { return y_min_; }
  float YMax() { return y_max_; }
  float ZMin() { return z_min_; }
  float ZMax() { return z_max_; }

 private:
  float x_res_;
  float x_min_;
  float x_max_;
  float y_res_;
  float y_min_;
  float y_max_;
  float z_res_;
  float z_min_;
  float z_max_;

  int x_size_;
  int y_size_;
  int z_size_;
  int size_;
};