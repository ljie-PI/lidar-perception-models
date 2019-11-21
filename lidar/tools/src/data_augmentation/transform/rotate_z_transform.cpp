#include <cmath>

#include <pcl/common/transforms.h>

#include "rotate_z_transform.h"

RotateZTransform::RotateZTransform(double ratio, double min_rot_angle, double max_rot_angle)
    : Transform(ratio) {
  name_ = "rotate_z";
  ang_rand_ptr_ = std::make_shared<UniformDistRandom>(min_rot_angle, max_rot_angle);
}

bool RotateZTransform::Apply(const Example &ori_example, Example *trans_example) {
  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
  double theta = ang_rand_ptr_->Generate();
  trans_mat(0, 0) = std::cos(theta);
  trans_mat(0, 1) = -std::sin(theta);
  trans_mat(1, 0) = std::sin(theta);
  trans_mat(1, 1) = std::cos(theta);

  const pcl::PointCloud<pcl::PointXYZI> &ori_pc = ori_example.GetPointCloud();
  pcl::PointCloud<pcl::PointXYZI> trans_pc;
  pcl::transformPointCloud(ori_pc, trans_pc, trans_mat);
  trans_example->SetPointCloud(trans_pc);

  const Label &ori_label = ori_example.GetLabel();
  const std::vector<BoundingBox> &ori_bboxes = ori_label.BoundingBoxes();
  std::vector<BoundingBox> trans_bboxes;
  for (const auto &ori_bbox : ori_bboxes) {
    Eigen::Vector4f ori_center(ori_bbox.center_x, ori_bbox.center_y, ori_bbox.center_z, 1);
    Eigen::Vector4f trans_center = trans_mat * ori_center;
    BoundingBox bbox{
        trans_center(0),
        trans_center(1),
        trans_center(2),
        ori_bbox.length,
        ori_bbox.width,
        ori_bbox.height,
        ori_bbox.heading + theta,
        ori_bbox.type
    };
    trans_bboxes.emplace_back(bbox);
  }
  Label trans_label(trans_bboxes);
  trans_example->SetLabel(trans_label);
  trans_example->SetValid(true);
  return true;
}
