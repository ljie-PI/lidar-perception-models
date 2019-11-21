#include <cmath>

#include "down_sample_transform.h"

DownSampleTransform::DownSampleTransform(double ratio, double range, double factor)
    : ObstacleTransform(ratio) {
  name_ = "down_sample";
  down_sample_range_ = range;
  down_sample_factor_ = factor;
  sample_rand_ = std::make_shared<UniformDistRandom>(0.0, 1.0);
}

bool DownSampleTransform::ApplyToObstacle(const pcl::PointCloud<pcl::PointXYZI> &ori_obs_pc,
                                          const Label &ori_label,
                                          pcl::PointCloud<pcl::PointXYZI> *trans_obs_pc,
                                          Label *trans_label,
                                          int obs_id) {
  BoundingBox bbox = ori_label.BoundingBoxes()[obs_id];
  float distance = std::sqrt(bbox.center_x * bbox.center_x +
      bbox.center_y * bbox.center_y + bbox.center_z * bbox.center_z);
  if (distance > down_sample_range_) {
    for (int i = 0; i < ori_obs_pc.size(); ++i) {
      trans_obs_pc->push_back(ori_obs_pc[i]);
    }
  } else {
    for (int i = 0; i < ori_obs_pc.size(); ++i) {
      if (sample_rand_->Generate() < down_sample_factor_) {
        trans_obs_pc->push_back(ori_obs_pc[i]);
      }
    }
  }
  trans_label->ModBoundingBox(obs_id, bbox);
  return true;
}

