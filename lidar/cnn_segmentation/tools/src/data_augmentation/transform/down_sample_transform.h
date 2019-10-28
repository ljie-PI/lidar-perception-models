#pragma once

#include "common/random.h"
#include "obstacle_transform.h"

class DownSampleTransform : public ObstacleTransform {
public:
  DownSampleTransform(double ratio, double range, double factor);

  ~DownSampleTransform() = default;

  bool ApplyToObstacle(const pcl::PointCloud<pcl::PointXYZI> &ori_obs_pc,
                       const Label &ori_label,
                       pcl::PointCloud<pcl::PointXYZI> *trans_obs_pc,
                       Label *trans_label,
                       int obs_id) override;

private:
  double down_sample_range_;
  double down_sample_factor_;

  std::shared_ptr<UniformDistRandom> sample_rand_;
};