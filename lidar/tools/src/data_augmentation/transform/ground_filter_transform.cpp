#include <iostream>
#include <pcl/common/transforms.h>

#include "ground_filter_transform.h"

GroundFilterTransform::GroundFilterTransform(double ratio, double ground_height)
    : Transform(ratio) {
  name_ = "ground_filter";
  ground_height_ = ground_height;
}

bool GroundFilterTransform::Apply(const Example &ori_example, Example *trans_example) {
  const pcl::PointCloud<pcl::PointXYZI> &ori_pc = ori_example.GetPointCloud();
  pcl::PointCloud<pcl::PointXYZI> trans_pc;
  for (int i = 0; i < ori_pc.size(); ++i) {
    if (ori_pc[i].z < ground_height_) {
      continue;
    }
    trans_pc.push_back(ori_pc[i]);
  }
  trans_example->SetPointCloud(trans_pc);

  const Label &ori_label = ori_example.GetLabel();
  Label trans_label;
  trans_label.DeepCopy(ori_label);
  trans_example->SetLabel(trans_label);
  trans_example->SetValid(true);
  return true;
}
