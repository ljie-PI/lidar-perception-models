#pragma once

#include "common/random.h"
#include "common/box2d.h"
#include "common/pcl_defs.h"
#include "common/label.h"
#include "voxel_mapping.h"
#include "pp_config.pb.h"
#include "pp_example.pb.h"

class TargetAssigner {
 public:
  explicit TargetAssigner(const pointpillars::PointPillarsConfig& config);
  ~TargetAssigner() = default;

  bool Assign(const LidarPointCloud& point_cloud, const Label& label,
              pointpillars::Example* example);

 private:
  bool GenerateAnchors(const LidarPointCloud& point_cloud,
                       std::vector<pointpillars::Anchor>* anchors);

  bool GenerateRpnLabels(const Label& label, std::vector<pointpillars::Label>* pp_labels);

  bool MatchAnchorLabel(const std::vector<pointpillars::Anchor>& anchors,
                        const std::vector<pointpillars::Label>& pp_labels,
                        pointpillars::Example* example);

  void AccumulateOccupy(const LidarPointCloud& point_cloud, int* voxel_occupy_acc);

  bool AnchorIsEmpty(float center_x, float center_y,
                     float length, float width, int* voxel_occupy_acc);

  float CalculateMatchScore(Box2D& anchor_box2d, Box2D& label_box2d);

  float match_thr_;
  float unmatch_thr_;

  std::vector<pointpillars::AnchorSize> anchor_sizes_;
  int anchor_size_cnt_;

  float sample_unmatch_ratio_;
  std::shared_ptr<UniformDistRandom> unmatch_anchor_sample_random_;

  std::shared_ptr<VoxelMapping> voxel_mapping_;

  friend class TargetAssignerTest_voxel_occupy_test_Test;
  friend class TargetAssignerTest_assign_test_Test;
};
