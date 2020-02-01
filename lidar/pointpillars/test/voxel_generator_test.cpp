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
  EXPECT_EQ(pointpillars::RANDOM, voxel_generator_->voxel_select_method_);
}

TEST_F(VoxelGeneratorTest, generate_voxel_test) {
  LidarPointCloud point_cloud;
  std::string full_pc_path = FLAGS_test_base + "/test_point_cloud1.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path, &point_cloud));

  pointpillars::Example pp_example;
  voxel_generator_->Generate(point_cloud, &pp_example);

  const auto& voxel = pp_example.voxel();
  EXPECT_EQ(9, voxel.num_voxel());
  EXPECT_EQ(7, voxel.feature_dim());
  EXPECT_EQ(399, voxel.data_size());
  // check the 4th voxel
  int voffset = 154;
  EXPECT_NEAR(0.21, voxel.data(voffset++), 1e-5);   // x
  EXPECT_NEAR(0.31, voxel.data(voffset++), 1e-5);   // y
  EXPECT_NEAR(0.1, voxel.data(voffset++), 1e-5);    // z
  EXPECT_NEAR(0.37443, voxel.data(voffset++), 1e-5);// distance
  EXPECT_NEAR(-0.04, voxel.data(voffset++), 1e-5);  // x to pillar center
  EXPECT_NEAR(-0.04, voxel.data(voffset++), 1e-5);  // y to pillar center
  EXPECT_NEAR(-0.9, voxel.data(voffset++), 1e-5);   // z to pillar center
  voffset = 189;
  EXPECT_NEAR(0.23, voxel.data(voffset++), 1e-5);   // x
  EXPECT_NEAR(0.30, voxel.data(voffset++), 1e-5);   // y
  EXPECT_NEAR(1.3, voxel.data(voffset++), 1e-5);    // z
  EXPECT_NEAR(0.37802, voxel.data(voffset++), 1e-5);// distance
  EXPECT_NEAR(-0.02, voxel.data(voffset++), 1e-5);  // x to pillar center
  EXPECT_NEAR(-0.05, voxel.data(voffset++), 1e-5);  // y to pillar center
  EXPECT_NEAR(0.3, voxel.data(voffset++), 1e-5);    // z to pillar center

  const auto& voxel_coord = pp_example.voxel_coord();
  EXPECT_EQ(9, voxel_coord.num_voxel());
  EXPECT_EQ(3, voxel_coord.coord_dim());
  EXPECT_EQ(27, voxel_coord.data_size());
  // check the 4th voxel
  voffset = 9;
  EXPECT_EQ(302, voxel_coord.data(voffset++));
  EXPECT_EQ(303, voxel_coord.data(voffset++));
  EXPECT_EQ(0, voxel_coord.data(voffset++));

  const auto& voxel_points = pp_example.voxel_points();
  EXPECT_EQ(9, voxel_points.num_voxel());
  EXPECT_EQ(9, voxel_points.data_size());
  EXPECT_EQ(10, voxel_points.data(0));
  EXPECT_EQ(6, voxel_points.data(1));
  EXPECT_EQ(5, voxel_points.data(4));
}

TEST_F(VoxelGeneratorTest, select_voxel_test) {
  LidarPointCloud point_cloud1;
  std::string full_pc_path1 = FLAGS_test_base + "/test_point_cloud2.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path1, &point_cloud1));

  pointpillars::Example pp_example1;
  voxel_generator_->Generate(point_cloud1, &pp_example1);
  const auto& voxel = pp_example1.voxel();
  EXPECT_EQ(10, voxel.num_voxel());
  EXPECT_EQ(7, voxel.feature_dim());
  EXPECT_EQ(70, voxel.data_size());

  LidarPointCloud point_cloud2;
  std::string full_pc_path2 = FLAGS_test_base + "/test_point_cloud3.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path2, &point_cloud2));

  pointpillars::Example pp_example2;
  voxel_generator_->voxel_select_method_ = pointpillars::BY_COUNT;
  voxel_generator_->Generate(point_cloud2, &pp_example2);
  voxel_generator_->voxel_select_method_ = pointpillars::RANDOM;
  const auto& voxel_points = pp_example2.voxel_points();
  for (int i = 0; i < 9; ++i) {
    EXPECT_TRUE(voxel_points.data(i) > 1);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
