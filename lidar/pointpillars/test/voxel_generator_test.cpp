#include "preprocess/voxel_generator.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <gflags/gflags.h>

#include "common/protobuf_util.h"
#include "test_utils.h"

DEFINE_string(test_base, "../lidar/pointpillars/test/test_base", "directory containing test data and test configs");

class VoxelGeneratorTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    pointpillars::PointPillarsConfig pp_config;
    std::string config_file = FLAGS_test_base + "/test_config.pb.txt";
    ProtobufUtil::ParseFromASCIIFile(config_file, &pp_config);
    voxel_generator_ = std::make_shared<VoxelGenerator>(pp_config);
  }

 protected:
  std::shared_ptr<VoxelGenerator> voxel_generator_;
};

TEST_F(VoxelGeneratorTest, read_config_test) {
  EXPECT_EQ(10, voxel_generator_->num_voxels_);
  EXPECT_EQ(10, voxel_generator_->num_points_per_voxel_);
  EXPECT_EQ(false, voxel_generator_->save_points_);
  EXPECT_EQ(false, voxel_generator_->log_voxel_num_);
  EXPECT_EQ(false, voxel_generator_->log_point_num_);
  EXPECT_EQ(false, voxel_generator_->use_reflection_);
}

TEST_F(VoxelGeneratorTest, generate_voxel_test) {
  LidarPointCloud point_cloud;
  std::string full_pc_path = FLAGS_test_base + "/test_point_cloud1.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path, &point_cloud));

  pointpillars::Example pp_example;
  voxel_generator_->Generate(point_cloud, &pp_example);

  const auto& voxel = pp_example.voxel();
  EXPECT_EQ(3, voxel.shape_size());
  EXPECT_EQ(9, voxel.shape(0));
  EXPECT_EQ(10, voxel.shape(1));
  EXPECT_EQ(7, voxel.shape(2));
  EXPECT_EQ(630, voxel.data_size());
  // check the 4th voxel
  int voffset = 210;
  EXPECT_NEAR(0.21, voxel.data(voffset++), 1e-5);   // x
  EXPECT_NEAR(0.31, voxel.data(voffset++), 1e-5);   // y
  EXPECT_NEAR(0.0, voxel.data(voffset++), 1e-5);    // z
  EXPECT_NEAR(0.37443, voxel.data(voffset++), 1e-5);// distance
  EXPECT_NEAR(-0.04, voxel.data(voffset++), 1e-5);  // x to pillar center
  EXPECT_NEAR(-0.04, voxel.data(voffset++), 1e-5);  // y to pillar center
  EXPECT_NEAR(-1.0, voxel.data(voffset++), 1e-5);   // z to pillar center

  const auto& voxel_coord = pp_example.voxel_coord();
  EXPECT_EQ(2, voxel_coord.shape_size());
  EXPECT_EQ(9, voxel_coord.shape(0));
  EXPECT_EQ(3, voxel_coord.shape(1));
  EXPECT_EQ(27, voxel_coord.data_size());
  // check the 4th voxel
  voffset = 9;
  EXPECT_EQ(302, voxel_coord.data(voffset++));
  EXPECT_EQ(303, voxel_coord.data(voffset++));
  EXPECT_EQ(0, voxel_coord.data(voffset++));

  const auto& voxel_points = pp_example.voxel_points();
  EXPECT_EQ(1, voxel_points.shape_size());
  EXPECT_EQ(9, voxel_points.shape(0));
  EXPECT_EQ(9, voxel_points.data_size());
  EXPECT_EQ(10, voxel_points.data(0));
  EXPECT_EQ(6, voxel_points.data(1));
  EXPECT_EQ(5, voxel_points.data(4));
}

TEST_F(VoxelGeneratorTest, sample_voxel_test) {
  LidarPointCloud point_cloud;
  std::string full_pc_path = FLAGS_test_base + "/test_point_cloud2.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path, &point_cloud));

  pointpillars::Example pp_example;
  voxel_generator_->Generate(point_cloud, &pp_example);

  const auto& voxel = pp_example.voxel();
  EXPECT_EQ(3, voxel.shape_size());
  EXPECT_EQ(10, voxel.shape(0));
  EXPECT_EQ(10, voxel.shape(1));
  EXPECT_EQ(7, voxel.shape(2));
  EXPECT_EQ(700, voxel.data_size());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}