#include <pcl/common/transforms.h>

#include "reflect_x_transform.h"

ReflectXTransform::ReflectXTransform(double ratio): Transform(ratio) {
  name_ = "reflect_x";
}

bool ReflectXTransform::Apply(const Example &ori_example, Example *trans_example) {
  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
  trans_mat(0, 0) = -1;

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
    double trans_heading = ori_bbox.heading > 0 ? M_PI - ori_bbox.heading : -M_PI - ori_bbox.heading;
    BoundingBox bbox{
        trans_center(0),
        trans_center(1),
        trans_center(2),
        ori_bbox.length,
        ori_bbox.width,
        ori_bbox.height,
        trans_heading,
        ori_bbox.type
    };
    trans_bboxes.emplace_back(bbox);
  }
  Label trans_label(trans_bboxes);
  trans_example->SetLabel(trans_label);
  trans_example->SetValid(true);
  return true;
}
