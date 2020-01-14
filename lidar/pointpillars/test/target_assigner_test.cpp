#include "preprocess/target_assigner.h"

#include <gtest/gtest.h>
#include <gflags/gflags.h>

#include "preprocess/voxel_generator.h"
#include "common/protobuf_util.h"
#include "test_utils.h"

DEFINE_string(test_base, "../lidar/pointpillars/test/test_base", "directory containing test data and test configs");

class TargetAssignerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    pointpillars::PointPillarsConfig pp_config;
    std::string config_file = FLAGS_test_base + "/test_config.pb.txt";
    ProtobufUtil::ParseFromASCIIFile(config_file, &pp_config);
    voxel_generator_ = std::make_shared<VoxelGenerator>(pp_config);
    target_assigner_ = std::make_shared<TargetAssigner>(pp_config);
  }

 protected:
  std::shared_ptr<VoxelGenerator> voxel_generator_;
  std::shared_ptr<TargetAssigner> target_assigner_;
};

TEST_F(TargetAssignerTest, voxel_occupy_test) {
  LidarPointCloud point_cloud;
  std::string full_pc_path = FLAGS_test_base + "/test_point_cloud1.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path, &point_cloud));

  int x_size = target_assigner_->voxel_mapping_->XSize();
  int y_size = target_assigner_->voxel_mapping_->YSize();
  int* voxel_occupy_acc = new int[x_size * y_size]();
  target_assigner_->AccumulateOccupy(point_cloud, voxel_occupy_acc);
  EXPECT_EQ(0, voxel_occupy_acc[Offset2D(50, 50, x_size, y_size)]);
  EXPECT_EQ(0, voxel_occupy_acc[Offset2D(300, 301, x_size, y_size)]);
  EXPECT_EQ(0, voxel_occupy_acc[Offset2D(299, 302, x_size, y_size)]);
  EXPECT_EQ(1, voxel_occupy_acc[Offset2D(300, 302, x_size, y_size)]);
  EXPECT_EQ(2, voxel_occupy_acc[Offset2D(301, 302, x_size, y_size)]);
  EXPECT_EQ(2, voxel_occupy_acc[Offset2D(302, 302, x_size, y_size)]);
  EXPECT_EQ(2, voxel_occupy_acc[Offset2D(303, 302, x_size, y_size)]);
  EXPECT_EQ(3, voxel_occupy_acc[Offset2D(304, 302, x_size, y_size)]);
  EXPECT_EQ(4, voxel_occupy_acc[Offset2D(305, 302, x_size, y_size)]);
  EXPECT_EQ(5, voxel_occupy_acc[Offset2D(306, 302, x_size, y_size)]);
  EXPECT_EQ(1, voxel_occupy_acc[Offset2D(300, 303, x_size, y_size)]);
  EXPECT_EQ(3, voxel_occupy_acc[Offset2D(301, 303, x_size, y_size)]);
  EXPECT_EQ(3, voxel_occupy_acc[Offset2D(301, 304, x_size, y_size)]);
  EXPECT_EQ(4, voxel_occupy_acc[Offset2D(302, 303, x_size, y_size)]);
  EXPECT_EQ(4, voxel_occupy_acc[Offset2D(303, 303, x_size, y_size)]);
  EXPECT_EQ(6, voxel_occupy_acc[Offset2D(304, 303, x_size, y_size)]);
  EXPECT_EQ(6, voxel_occupy_acc[Offset2D(304, 304, x_size, y_size)]);
  EXPECT_EQ(8, voxel_occupy_acc[Offset2D(305, 303, x_size, y_size)]);
  EXPECT_EQ(9, voxel_occupy_acc[Offset2D(306, 303, x_size, y_size)]);

  EXPECT_TRUE(target_assigner_->AnchorIsEmpty(0.35, 0.45, 0.28, 0.08, voxel_occupy_acc));
  EXPECT_FALSE(target_assigner_->AnchorIsEmpty(0.35, 0.45, 0.28, 0.28, voxel_occupy_acc));
  EXPECT_FALSE(target_assigner_->AnchorIsEmpty(0.35, 0.25, 0.28, 0.08, voxel_occupy_acc));
  EXPECT_FALSE(target_assigner_->AnchorIsEmpty(0.75, 0.25, 0.28, 0.08, voxel_occupy_acc));

  delete [] voxel_occupy_acc;
}

TEST_F(TargetAssignerTest, assign_test) {
  LidarPointCloud point_cloud;
  std::string full_pc_path = FLAGS_test_base + "/test_point_cloud1.txt";
  EXPECT_EQ(true, LoadPointCloud(full_pc_path, &point_cloud));
  std::string full_label_path = FLAGS_test_base + "/test_label.txt";
  Label label;
  EXPECT_EQ(true, label.FromFile(full_label_path));

  pointpillars::Example pp_example;
  voxel_generator_->Generate(point_cloud, &pp_example);
  target_assigner_->Assign(point_cloud, label, &pp_example);
  std::vector<pointpillars::Anchor> pos_anchors;
  for (int i = 0; i < pp_example.anchor_size(); ++i) {
    auto& anchor = pp_example.anchor(i);
    if (anchor.is_postive()) {
      pos_anchors.push_back(anchor);
      std::cout << "anchor " << i << ": "
                << ", center_x = " << anchor.center_x()
                << ", center_y = " << anchor.center_y()
                << ", length = " << anchor.length()
                << ", width = " << anchor.width()
                << ", rotation = " << anchor.rotation()
                << ", target_label = " << anchor.target_label() << std::endl;
    }
  }
  EXPECT_EQ(4, pos_anchors.size());
  for (auto& pos_anchor : pos_anchors) {
    EXPECT_TRUE((DoubleNear(pos_anchor.center_x(), 0.55) && DoubleNear(pos_anchor.center_y(), 0.35) && pos_anchor.target_label() == 1)
             || (DoubleNear(pos_anchor.center_x(), 0.55) && DoubleNear(pos_anchor.center_y(), 0.25) && pos_anchor.target_label() == 1)
             || (DoubleNear(pos_anchor.center_x(), 0.15) && DoubleNear(pos_anchor.center_y(), 0.25) && pos_anchor.target_label() == 0));
  }

  for (int i = 0; i < pp_example.label_size(); ++i) {
    auto& label = pp_example.label(i);
    std::cout << "label " << label.label_id() << ": "
              << ", center_x = " << label.center_x()
              << ", center_y = " << label.center_y()
              << ", center_z = " << label.center_x()
              << ", length = " << label.length()
              << ", width = " << label.width()
              << ", height = " << label.height()
              << ", yaw = " << label.yaw()
              << ", type = " << label.type() << std::endl;
    EXPECT_TRUE((i == 0 && label.type() == 1) || (i == 1 && label.type() == 4));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
