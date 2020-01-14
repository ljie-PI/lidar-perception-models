#include "preprocess/voxel_mapping.h"

#include <gtest/gtest.h>

class VoxelMappingTest: public ::testing::Test {
 protected:
  virtual void SetUp() {
    voxel_mapping_ = std::make_shared<VoxelMapping>(
        0.15, 0.0, 30.0, 0.10, -15.0, 14.98, 2.5, -2.5, 2.5);
  }

 protected:
  std::shared_ptr<VoxelMapping> voxel_mapping_;
};

TEST_F(VoxelMappingTest, index_offset_convert_test) {
  int x_idx = -1;
  int y_idx = 1000;
  int z_idx = 1;
  EXPECT_EQ(-1, voxel_mapping_->VoxelIndexToOffset(x_idx, y_idx, z_idx));
  x_idx = 50;
  EXPECT_EQ(-1, voxel_mapping_->VoxelIndexToOffset(x_idx, y_idx, z_idx));
  z_idx = 1;
  EXPECT_EQ(-1, voxel_mapping_->VoxelIndexToOffset(x_idx, y_idx, z_idx));
  y_idx = 20;
  EXPECT_EQ(30041, voxel_mapping_->VoxelIndexToOffset(x_idx, y_idx, z_idx));

  int offset = -1;
  EXPECT_EQ(false, voxel_mapping_->OffsetToVoxelIndex(offset, &x_idx, &y_idx, &z_idx));
  offset = 120000;
  EXPECT_EQ(false, voxel_mapping_->OffsetToVoxelIndex(offset, &x_idx, &y_idx, &z_idx));
  offset = 119999;
  EXPECT_EQ(true, voxel_mapping_->OffsetToVoxelIndex(offset, &x_idx, &y_idx, &z_idx));
  EXPECT_EQ(199, x_idx);
  EXPECT_EQ(299, y_idx);
  EXPECT_EQ(1, z_idx);
  offset = 30041;
  EXPECT_EQ(true, voxel_mapping_->OffsetToVoxelIndex(offset, &x_idx, &y_idx, &z_idx));
  EXPECT_EQ(50, x_idx);
  EXPECT_EQ(20, y_idx);
  EXPECT_EQ(1, z_idx);
}

TEST_F(VoxelMappingTest, pos_mapping_test) {
  float x = -1.0;
  float y = 15.0;
  float z = 2.5;
  int x_idx;
  int y_idx;
  int z_idx;
  EXPECT_EQ(-1, voxel_mapping_->MapToOffset(x, y, z));
  EXPECT_EQ(false, voxel_mapping_->MapToVoxelIndex(x, y, z, &x_idx, &y_idx, &z_idx));
  x = 9;
  EXPECT_EQ(false, voxel_mapping_->MapToVoxelIndex(x, y, z, &x_idx, &y_idx, &z_idx));
  EXPECT_EQ(-1, voxel_mapping_->MapToOffset(x, y, z));
  y = 8.05;
  EXPECT_EQ(false, voxel_mapping_->MapToVoxelIndex(x, y, z, &x_idx, &y_idx, &z_idx));
  EXPECT_EQ(-1, voxel_mapping_->MapToOffset(x, y, z));
  z = 2.499999;
  EXPECT_EQ(true, voxel_mapping_->MapToVoxelIndex(x, y, z, &x_idx, &y_idx, &z_idx));
  EXPECT_EQ(59, x_idx);
  EXPECT_EQ(230, y_idx);
  EXPECT_EQ(1, z_idx);
  EXPECT_EQ(35861, voxel_mapping_->MapToOffset(x, y, z));
}

TEST_F(VoxelMappingTest, voxle_center_test) {
  int x_idx = -1;
  int y_idx = 1000;
  int z_idx = 1;
  float center_x;
  float center_y;
  float center_z;
  EXPECT_EQ(false, voxel_mapping_->VoxelCenter(x_idx, y_idx, z_idx, &center_x, &center_y, &center_z));
  x_idx = 50;
  EXPECT_EQ(false, voxel_mapping_->VoxelCenter(x_idx, y_idx, z_idx, &center_x, &center_y, &center_z));
  z_idx = 0;
  EXPECT_EQ(false, voxel_mapping_->VoxelCenter(x_idx, y_idx, z_idx, &center_x, &center_y, &center_z));
  y_idx = 20;
  EXPECT_EQ(true, voxel_mapping_->VoxelCenter(x_idx, y_idx, z_idx, &center_x, &center_y, &center_z));
  EXPECT_NEAR(7.575, center_x, 1e-5);
  EXPECT_NEAR(-12.95, center_y, 1e-5);
  EXPECT_NEAR(-1.25, center_z, 1e-5);

  int offset = -1;
  EXPECT_EQ(false, voxel_mapping_->VoxelCenter(offset, &center_x, &center_y, &center_z));
  offset = 120000;
  EXPECT_EQ(false, voxel_mapping_->VoxelCenter(offset, &center_x, &center_y, &center_z));
  offset = 30041;
  EXPECT_EQ(true, voxel_mapping_->VoxelCenter(offset, &center_x, &center_y, &center_z));
  EXPECT_NEAR(7.575, center_x, 1e-5);
  EXPECT_NEAR(-12.95, center_y, 1e-5);
  EXPECT_NEAR(1.25, center_z, 1e-5);
}
