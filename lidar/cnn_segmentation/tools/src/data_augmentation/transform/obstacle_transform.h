#pragma once

#include "transform.h"

class ObstacleTransform : public Transform {
public:
  ObstacleTransform(double ratio);

  ~ObstacleTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;

  virtual bool ApplyToObstacle(const pcl::PointCloud<pcl::PointXYZI> &ori_obs_pc,
                               const Label &ori_label,
                               pcl::PointCloud<pcl::PointXYZI> *trans_obs_pc,
                               Label *trans_label,
                               int obs_id) = 0;

private:
  bool CollectObstacles(const pcl::PointCloud<pcl::PointXYZI> &pc, const Label &label);

  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obs_pc_ptrs_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr bg_pc_ptr_;
};