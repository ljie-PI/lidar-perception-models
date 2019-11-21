#include <pcl/common/transforms.h>

#include "move_transform.h"

MoveTransform::MoveTransform(double ratio, double move_mean, double move_std)
    : Transform(ratio) {
  name_ = "move";
  x_rand_ptr_ = std::make_shared<NormalDistRandom>(move_mean, move_std);
  y_rand_ptr_ = std::make_shared<NormalDistRandom>(move_mean, move_std);
  z_rand_ptr_ = std::make_shared<NormalDistRandom>(move_mean, move_std);
}

bool MoveTransform::Apply(const Example &ori_example, Example *trans_example) {
  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
  trans_mat(0, 3) = x_rand_ptr_->Generate();
  trans_mat(1, 3) = y_rand_ptr_->Generate();
  trans_mat(2, 3) = z_rand_ptr_->Generate();

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
        ori_bbox.heading,
        ori_bbox.type
    };
    trans_bboxes.emplace_back(bbox);
  }
  Label trans_label(trans_bboxes);
  trans_example->SetLabel(trans_label);
  trans_example->SetValid(true);
  return true;
}
