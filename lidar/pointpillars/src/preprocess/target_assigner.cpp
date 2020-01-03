#include "target_assigner.h"

#include <cmath>
#include <algorithm>

TargetAssigner::TargetAssigner(const pointpillars::PointPillarsConfig& config) {
  match_thr_ = config.anchor_config().match_thr();
  unmatch_thr_ = config.anchor_config().unmatch_thr();
  output_factor_ = config.anchor_config().output_factor();

  anchor_size_cnt_ = config.anchor_config().anchor_size_size();
  for (int i = 0; i < anchor_size_cnt_; ++i) {
    anchor_sizes_.push_back(config.anchor_config().anchor_size(i));
  }

  sample_unmatch_anchor_ = config.anchor_config().sample_unmatch_anchor();
  sample_unmatch_ratio_ = config.anchor_config().sample_unmatch_ratio();
  unmatch_anchor_sample_random_ = std::make_shared<UniformDistRandom>(0.0, 1.0);

  float x_res = config.voxel_config().x_resolution() * output_factor_;
  float y_res = config.voxel_config().y_resolution() * output_factor_;
  float z_res = config.voxel_config().z_resolution() * output_factor_;
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
}

bool TargetAssigner::Assign(const LidarPointCloud &point_cloud, const Label &label,
                            pointpillars::Example *example) {
  std::vector<pointpillars::Anchor> anchors;
  if (!GenerateAnchors(point_cloud, &anchors)) {
    std::cerr << "Failed to generate anchors!" << std::endl;
    return false;
  }

  std::vector<pointpillars::Label> pp_labels;
  if (!GenerateRpnLabels(label, &pp_labels)) {
    std::cerr << "Failed to generate RPN labels!" << std::endl;
    return false;
  }

  
  if (!MatchAnchorLabel(anchors, pp_labels, example)) {
    std::cerr << "Failed to match anchors and RPN labels to produce learning targets!" << std::endl;
    return false;
  }

  return true;
}

 /**
  * each element in voxel_occupy_acc e(i,j) is count of non-empty voxels in the region
  * from (0, 0)(as left top) to (i, j) (as right bottom)
  */
void TargetAssigner::AccumulateOccupy(const LidarPointCloud& point_cloud, int* voxel_occupy_acc) {
  size_t pc_size = point_cloud.size();
  for (size_t pt_id = 0; pt_id < pc_size; ++pt_id) {
    const LidarPoint& point = point_cloud.at(pt_id);
    int x_offset, y_offset, z_offset;
    if (!voxel_mapping_->MapToVoxelIndex(point.x, point.y, point.z,
                                         &x_offset, &y_offset, &z_offset)) {
      // point out of range, will ignore
      continue;
    }
    int xy_offset = y_offset * voxel_mapping_->XSize() + x_offset;
    voxel_occupy_acc[xy_offset] = 1;
  }

  for (int y_offset = 0; y_offset < voxel_mapping_->YSize(); ++y_offset) {
    for (int x_offset = 0; x_offset < voxel_mapping_->XSize(); ++x_offset) {
      int xy_offset = y_offset * voxel_mapping_->XSize() + x_offset;
      if (x_offset > 0) {
        int left_offset = xy_offset - 1;
        voxel_occupy_acc[xy_offset] += voxel_occupy_acc[left_offset];
      }
      if (y_offset > 0) {
        int up_offset = (y_offset - 1) * voxel_mapping_->XSize() + x_offset;
        voxel_occupy_acc[xy_offset] += voxel_occupy_acc[up_offset];
      }
      if (x_offset > 0 && y_offset > 0) {
        int left_up_offset = (y_offset - 1) * voxel_mapping_->XSize() + x_offset - 1;
        voxel_occupy_acc[xy_offset] -= voxel_occupy_acc[left_up_offset];
      }
    }
  }
}

bool TargetAssigner::AnchorIsEmpty(float center_x, float center_y,
                                   float length, float width,
                                   int* voxel_occupy_acc) {
  float min_x = center_x - length * 0.5f;
  min_x = std::max(min_x, voxel_mapping_->XMin());
  float max_x = center_x + length * 0.5f;
  max_x = std::min(max_x, voxel_mapping_->XMax());
  float min_y = center_y - width * 0.5f;
  min_y = std::max(min_y, voxel_mapping_->YMin());
  float max_y = center_y + width * 0.5f;
  max_y = std::min(max_y, voxel_mapping_->YMax());
  int min_x_idx, max_x_idx, min_z_idx, min_y_idx, max_y_idx, max_z_idx;
  voxel_mapping_->MapToVoxelIndex(min_x, min_y, 0, &min_x_idx, &min_y_idx, &min_z_idx);
  voxel_mapping_->MapToVoxelIndex(max_x, max_y, 0, &max_x_idx, &max_y_idx, &max_z_idx);

  int left_up_offset = min_y_idx * voxel_mapping_->XSize() + min_x_idx;
  int right_up_offset = min_y_idx * voxel_mapping_->XSize() + max_x_idx;
  int left_bottom_offset = max_y_idx * voxel_mapping_->XSize() + min_x_idx;
  int right_bottom_offset = max_y_idx * voxel_mapping_->XSize() + max_x_idx;

  return voxel_occupy_acc[right_bottom_offset] + voxel_occupy_acc[left_up_offset]
         - voxel_occupy_acc[right_up_offset] - voxel_occupy_acc[left_bottom_offset] > 0;
}

bool TargetAssigner::GenerateAnchors(const LidarPointCloud& point_cloud,
                                     std::vector<pointpillars::Anchor>* anchors) {
  // used to filter empty anchors
  int* voxel_occupy_acc = new int[voxel_mapping_->XSize() * voxel_mapping_->YSize()];
  AccumulateOccupy(point_cloud, voxel_occupy_acc);

  for (int y_offset = 0; y_offset < voxel_mapping_->YSize(); ++y_offset) {
    for (int x_offset = 0; x_offset < voxel_mapping_->XSize(); ++x_offset) {
      for (int size_idx = 0; size_idx < anchor_size_cnt_; ++size_idx) {
        float x_center, y_center, z_center;
        voxel_mapping_->VoxelCenter(x_offset, y_offset, 0, &x_center, &y_center, &z_center);
        const pointpillars::AnchorSize& anchor_size = anchor_sizes_[size_idx];

        // length matches x axis, width match y axis
        if (!AnchorIsEmpty(x_center, y_center,
                           anchor_size.length(), anchor_size.width(),
                           voxel_occupy_acc)) {
          pointpillars::Anchor anchor;
          anchor.set_center_x(x_center);
          anchor.set_center_y(y_center);
          anchor.set_length(anchor_sizes_[size_idx].length());
          anchor.set_width(anchor_sizes_[size_idx].width());
          anchor.set_height(anchor_sizes_[size_idx].height());
          anchor.set_rotation(0);
          anchors->emplace_back(anchor);
        }

        // length matches y axis, width match x axis
        if (!AnchorIsEmpty(x_center, y_center,
                           anchor_size.width(), anchor_size.length(),
                           voxel_occupy_acc)) {
          pointpillars::Anchor anchor;
          anchor.set_center_x(x_center);
          anchor.set_center_y(y_center);
          anchor.set_length(anchor_sizes_[size_idx].length());
          anchor.set_width(anchor_sizes_[size_idx].width());
          anchor.set_height(anchor_sizes_[size_idx].height());
          anchor.set_rotation(M_PI_2);
          anchors->emplace_back(anchor);
        }
      }
    }
  }
  delete [] voxel_occupy_acc;
  return true;
}

bool TargetAssigner::GenerateRpnLabels(const Label& label,
                                       std::vector<pointpillars::Label>* pp_labels) {
  int bbox_cnt = label.BoundingBoxCount();
  const std::vector<BoundingBox>& bboxes = label.BoundingBoxes();
  for (int lid = 0; lid < bbox_cnt; ++lid) {
    BoundingBox box = bboxes[lid];
    pointpillars::Label pp_label;
    pp_label.set_label_id(lid);
    pp_label.set_type(box.type);
    pp_label.set_center_x(static_cast<float>(box.center_x));
    pp_label.set_center_y(static_cast<float>(box.center_y));
    pp_label.set_center_z(static_cast<float>(box.center_z));
    pp_label.set_length(static_cast<float>(box.length));
    pp_label.set_width(static_cast<float>(box.width));
    pp_label.set_height(static_cast<float>(box.height));
    pp_label.set_yaw(static_cast<float>(box.height));
    pp_labels->emplace_back(pp_label);
  }
}

float TargetAssigner::CalculateMatchScore(const pointpillars::Anchor &anchor, Box2D &label_box2d) {
  Box2D anchor_box2d(anchor.center_x(), anchor.center_y(),
                     anchor.length(), anchor.width(), anchor.rotation());
  if (anchor_box2d.MaxX() < label_box2d.MinX() ||
      anchor_box2d.MinX() > label_box2d.MaxX() ||
      anchor_box2d.MaxY() < label_box2d.MinY() ||
      anchor_box2d.MinY() > label_box2d.MaxY()) {
    return 0.0f;
  }
  return static_cast<float>(anchor_box2d.IouWith(label_box2d));
}

/**
 * According to the paper:
We use the same anchors and matching strategy as [33].
Each class anchor is described by a width, length, height,
and z center, and is applied at two orientations: 0 and 90
degrees. Anchors are matched to ground truth using the 2D
IoU with the following rules. A positive match is either
the highest with a ground truth box, or above the positive
match threshold, while a negative match is below the negative threshold.
All other anchors are ignored in the loss
 */
bool TargetAssigner::MatchAnchorLabel(const std::vector<pointpillars::Anchor>& anchors,
                                      const std::vector<pointpillars::Label>& pp_labels,
                                      pointpillars::Example* example) {
  // anchors didn't match any labels
  std::vector<int> unmatch_anchors_ids;

  // anchor id mostly match each label
  std::vector<int> max_match_anchor_of_labels(pp_labels.size(), -1);
  // max match score of each label
  std::vector<float> max_match_score_of_labels(pp_labels.size(), 0.0);

  // label id mostly match each anchor
  std::vector<int> max_match_label_of_anchors;
  max_match_label_of_anchors.reserve(pp_labels.size() * 5);
  // max match score of each anchor
  std::vector<float> max_match_score_of_anchors;
  max_match_score_of_anchors.reserve(pp_labels.size() * 5);

  // cache label boxes and use min-max bound to filter non-overlap <anchor, label> pairs
  std::vector<Box2D> label_box2ds;
  for (const auto & label : pp_labels) {
    Box2D label_box2d(label.center_x(), label.center_y(),
                      label.length(), label.width(), label.yaw());
    label_box2ds.emplace_back(label_box2d);
  }

  for (int anch_id = 0; anch_id < anchors.size(); ++anch_id) {
    const pointpillars::Anchor& anchor = anchors[anch_id];
    float max_match_score_of_anchor = 0.0f;
    int max_match_label_of_anchor = -1;
    for (int lab_id = 0; lab_id < label_box2ds.size(); ++lab_id) {
      Box2D& label_box2d = label_box2ds[lab_id];
      float score = CalculateMatchScore(anchor, label_box2d);
      if (score > max_match_score_of_anchor) {
        max_match_score_of_anchor = score;
        max_match_label_of_anchor = lab_id;
      }
      if (score > max_match_score_of_labels[lab_id]) {
        max_match_score_of_labels[lab_id] = score;
        max_match_anchor_of_labels[lab_id] = anch_id;
      }
    }
    if (max_match_score_of_anchor == 0.0) {
      unmatch_anchors_ids.push_back(anch_id);
      continue;
    }
    max_match_label_of_anchors.push_back(max_match_label_of_anchor);
    max_match_score_of_anchors.push_back(max_match_score_of_anchor);
  }

  // anchors didn't match any labels
  for (auto unmatch_aid : unmatch_anchors_ids) {
    bool add = true;
    if (sample_unmatch_anchor_) {
      double randnum = unmatch_anchor_sample_random_->Generate();
      if (randnum >= sample_unmatch_ratio_) {
        add = false;
      }
    }
    if (add) {
      pointpillars::Anchor* anchor = example->add_anchor();
      anchor->CopyFrom(anchors[unmatch_aid]);
      anchor->set_target_label(-1);
      anchor->set_is_postive(false);
    }
  }

  // add anchors whose max match score above match_thr_ as postive anchors
  // add anchors whose max match score below unmatch_thr_ as negative anchors
  for (int anch_id = 0; anch_id < max_match_label_of_anchors.size(); ++anch_id) {
    float match_score = max_match_score_of_anchors[anch_id];
    if (match_score < match_thr_ && match_score >= unmatch_thr_) {
      // anchors whose max match score between unmatch_thr_ and match_thr will be ignored
      continue;
    }
    pointpillars::Anchor* anchor = example->add_anchor();
    anchor->CopyFrom(anchors[anch_id]);
    anchor->set_target_label(max_match_label_of_anchors[anch_id]);
    if (match_score >= match_thr_) {
      anchor->set_is_postive(true);
    } else {
      anchor->set_is_postive(false);
    }
  }

  // for each label, if no anchor has match score above match_thr_,
  // the anchor with max match score will be used as postive anchor
  for (int lab_id = 0; lab_id < max_match_anchor_of_labels.size(); ++lab_id) {
    int anch_id = max_match_anchor_of_labels[lab_id];
    float match_score = max_match_score_of_labels[lab_id];
    if (match_score > 0.0) {
      pointpillars::Anchor* anchor = example->add_anchor();
      anchor->CopyFrom(anchors[anch_id]);
      anchor->set_target_label(lab_id);
      anchor->set_is_postive(true);
    }
  }

  // add labels
  for (int lab_id = 0; lab_id < pp_labels.size(); ++lab_id) {
    pointpillars::Label* example_label = example->add_label();
    example_label->CopyFrom(pp_labels[lab_id]);
    example_label->set_label_id(lab_id);
  }

}
