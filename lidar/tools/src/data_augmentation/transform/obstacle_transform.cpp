#include <iostream>
#include <unordered_map>
#include <pcl/common/transforms.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

#include "obstacle_transform.h"

ObstacleTransform::ObstacleTransform(double ratio): Transform(ratio) {}

bool ObstacleTransform::Apply(const Example &ori_example, Example *trans_example) {
  CollectObstacles(ori_example.GetPointCloud(), ori_example.GetLabel());
  pcl::PointCloud<pcl::PointXYZI> trans_pc;
  Label trans_label;
  trans_label.SetBoundingBoxCount(ori_example.GetLabel().BoundingBoxCount());
  for (size_t i = 0; i < obs_pc_ptrs_.size(); ++i) {
    pcl::PointCloud<pcl::PointXYZI> &ori_obs_pc = *(obs_pc_ptrs_[i]);
    pcl::PointCloud<pcl::PointXYZI> trans_obs_pc;
    ApplyToObstacle(ori_obs_pc, ori_example.GetLabel(), &trans_obs_pc, &trans_label, i);
    for (int pid = 0; pid < trans_obs_pc.size(); ++pid) {
      trans_pc.push_back(trans_obs_pc[pid]);
    }
  }
  for (size_t i = 0; i < bg_pc_ptr_->size(); ++i) {
    trans_pc.push_back(bg_pc_ptr_->at(i));
  }
  trans_example->SetPointCloud(trans_pc);
  trans_example->SetLabel(trans_label);
  trans_example->SetValid(true);
  return true;
}

bool ObstacleTransform::CollectObstacles(const pcl::PointCloud<pcl::PointXYZI> &pc, const Label &label) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr(new pcl::PointCloud<pcl::PointXYZI>(pc));
  std::vector<BoundingBox> bboxes = label.BoundingBoxes();
  int obs_cnt = label.BoundingBoxCount();
  std::unordered_map<int, int> obs_point_map;
  for (size_t i = 0; i < obs_cnt; ++i) {
    auto &bbox = bboxes[i];
    pcl::PointCloud<pcl::PointXYZI> obs_pc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr bbox_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<Eigen::Vector4f> box_corners;
    bbox.GetCorners(&box_corners);
    for (auto &corner : box_corners) {
      pcl::PointXYZI point;
      point.x = corner.x();
      point.y = corner.y();
      point.z = corner.z();
      point.intensity = 1.0;
      bbox_pc_ptr->push_back(point);
    }

    pcl::ConvexHull<pcl::PointXYZI> hull;
    hull.setInputCloud(bbox_pc_ptr);
    hull.setDimension(3);
    pcl::PointCloud<pcl::PointXYZI>::Ptr surface_hull_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(*surface_hull_ptr, polygons);

    pcl::CropHull<pcl::PointXYZI> bb_filter;

    bb_filter.setDim(3);
    bb_filter.setInputCloud(pc_ptr);
    bb_filter.setHullIndices(polygons);
    bb_filter.setHullCloud(surface_hull_ptr);

    std::vector<int> obs_indices;
    bb_filter.setCropOutside(true);
    bb_filter.filter(obs_indices);
    for (int idx : obs_indices) {
      obs_point_map.insert(std::make_pair(idx, i));
    }
  }

  bg_pc_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  obs_pc_ptrs_.resize(obs_cnt);
  for (size_t i = 0; i < obs_cnt; ++i) {
    obs_pc_ptrs_[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
  }
  for (size_t i = 0; i < pc.size(); ++i) {
    pcl::PointXYZI point;
    point.x = pc[i].x;
    point.y = pc[i].y;
    point.z = pc[i].z;
    point.intensity = pc[i].intensity;
    auto map_iter = obs_point_map.find(i);
    if (map_iter == obs_point_map.end()) {
      bg_pc_ptr_->push_back(point);
    } else {
      int obs_id = map_iter->second;
      obs_pc_ptrs_[obs_id]->push_back(point);
    }
  }
}
