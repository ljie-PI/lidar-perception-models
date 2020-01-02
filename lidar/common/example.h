#pragma once

#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "lidar/common/label.h"

class Example {
public:
  Example();

  Example(pcl::PointCloud<pcl::PointXYZI> &point_cloud, Label &label);

  Example(std::string &example_id, std::string &pcd_file, std::string &label_file);

  ~Example() = default;

  const bool IsValid() const;
  void SetValid(bool is_valid);

  const std::string &ExampleID() const;
  void SetExampleID(const std::string &example_id);

  const std::string &PCDFile() const;

  const std::string &LabelFile() const;

  const pcl::PointCloud<pcl::PointXYZI> &GetPointCloud() const;
  void SetPointCloud(pcl::PointCloud<pcl::PointXYZI> &point_cloud);

  const Label &GetLabel() const;
  void SetLabel(Label &label);

private:
  bool is_valid_;
  std::string example_id_;
  std::string pcd_file_;
  std::string label_file_;
  pcl::PointCloud<pcl::PointXYZI> point_cloud_;
  Label label_;
};
