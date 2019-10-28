#include <cmath>

#include "up_sample_transform.h"

UpSampleTransform::UpSampleTransform(double ratio, double range, double factor)
    : ObstacleTransform(ratio) {
  name_ = "up_sample";
  up_sample_range_ = range;
  up_sample_factor_ = factor;
  sample_rand_ = std::make_shared<UniformDistRandom>(0.0, 1.0);
}

bool UpSampleTransform::ApplyToObstacle(const pcl::PointCloud<pcl::PointXYZI> &ori_obs_pc,
                                        const Label &ori_label,
                                        pcl::PointCloud<pcl::PointXYZI> *trans_obs_pc,
                                        Label *trans_label,
                                        int obs_id) {
  BoundingBox bbox = ori_label.BoundingBoxes()[obs_id];
  for (int i = 0; i < ori_obs_pc.size(); ++i) {
    trans_obs_pc->push_back(ori_obs_pc[i]);
  }
  float distance = std::sqrt(bbox.center_x * bbox.center_x +
                             bbox.center_y * bbox.center_y + bbox.center_z * bbox.center_z);
  if (distance > up_sample_range_) {
    int ori_size = ori_obs_pc.size();
    int up_sample_cnt = std::ceil(ori_size * (up_sample_factor_ - 1.0));
    for (int i = 0; i < up_sample_cnt; ++i) {
      int id1 = std::floor(sample_rand_->Generate() * ori_size);
      auto &point1 = ori_obs_pc[id1];
      int id2 = std::floor(sample_rand_->Generate() * ori_size);
      auto &point2 = ori_obs_pc[id2];
      pcl::PointXYZI new_point;
      new_point.x = (point1.x + point2.x) / 2.0;
      new_point.y = (point1.y + point2.y) / 2.0;
      new_point.z = (point1.z + point2.z) / 2.0;
      new_point.intensity = (point1.intensity + point2.intensity) / 2.0;
      trans_obs_pc->push_back(new_point);
    }
  }
  trans_label->ModBoundingBox(obs_id, bbox);
  return true;
}

