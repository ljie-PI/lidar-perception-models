#include <iostream>
#include <pcl/io/pcd_io.h>

#include "example.h"

Example::Example() {
  is_valid_ = false;
}

Example::Example(pcl::PointCloud<pcl::PointXYZI> &point_cloud, Label &label) {
  point_cloud_ = point_cloud;
  label_ = label;
}

Example::Example(std::string &example_id, std::string &pcd_file, std::string &label_file)
    : example_id_(example_id), pcd_file_(pcd_file), label_file_(label_file) {
  is_valid_ = false;
  if (pcl::io::loadPCDFile(pcd_file, point_cloud_) < 0) {
    std::cerr << "Failed to load pcd file: " << pcd_file << std::endl;
    return;
  }
  if (!label_.FromFile(label_file)) {
    std::cerr << "Failed to load label file: " << label_file << std::endl;
    return;
  }
  is_valid_ = true;
}

void Example::SetExampleID(const std::string &example_id) {
  example_id_ = example_id;
}

const bool Example::IsValid() const {
  return is_valid_;
}

void Example::SetValid(bool is_valid) {
  is_valid_ = is_valid;
}

const std::string &Example::ExampleID() const {
  return example_id_;
}

const std::string &Example::PCDFile() const {
  return pcd_file_;
}

const std::string &Example::LabelFile() const {
  return label_file_;
}

const pcl::PointCloud<pcl::PointXYZI> &Example::GetPointCloud() const {
  return point_cloud_;
}

void Example::SetPointCloud(pcl::PointCloud<pcl::PointXYZI> &point_cloud) {
  point_cloud_ = point_cloud;
}

const Label &Example::GetLabel() const {
  return label_;
}

void Example::SetLabel(Label &label) {
  label_ = label;
}
