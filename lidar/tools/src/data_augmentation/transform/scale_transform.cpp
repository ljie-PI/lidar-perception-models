#include <pcl/common/transforms.h>

#include "scale_transform.h"

ScaleTransform::ScaleTransform(double ratio, double min_scale, double max_scale)
    : Transform(ratio) {
  name_ = "scale";
  factor_rand_ptr_ = std::make_shared<UniformDistRandom>(min_scale, max_scale);
}

bool ScaleTransform::Apply(const Example &ori_example, Example *trans_example) {
  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
  double scale_factor = factor_rand_ptr_->Generate();
  trans_mat(0, 0) = scale_factor;
  trans_mat(1, 1) = scale_factor;
  trans_mat(2, 2) = scale_factor;

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
        ori_bbox.length * scale_factor,
        ori_bbox.width * scale_factor,
        ori_bbox.height * scale_factor,
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
