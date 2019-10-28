#include <iostream>

#include "common/file_util.h"
#include "transform.h"

Transform::Transform(double ratio) {
  trans_ratio_ = ratio;
  random_ptr_ = std::make_shared<UniformDistRandom>(0.0, 1.0);
}

std::string Transform::Name() {
  return name_;
}

bool Transform::ApplyByRatio(const Example &ori_example,
                             Example *trans_example) {
  double rand_num = random_ptr_->Generate();
  if (rand_num < trans_ratio_) {
    std::cout << Name() << " is applied" << std::endl;
    Apply(ori_example, trans_example);
    trans_example->SetExampleID(ori_example.ExampleID());
    trans_example->SetValid(true);
  } else {
    std::cout << Name() << " is skipped" << std::endl;
  }
}

bool Transform::Save(const Example &example,
                     const std::string &out_pcd_dir,
                     const std::string &out_label_dir) {
  std::string trans_pcd_dir = out_pcd_dir + "/" + Name();
  if (!FileUtil::Exists(trans_pcd_dir)) {
    FileUtil::Mkdir(trans_pcd_dir);
  }
  std::string pcd_file = trans_pcd_dir + "/" + example.ExampleID() + ".pcd";
  pcl::io::savePCDFileBinaryCompressed(pcd_file, example.GetPointCloud());
  std::cout << "Wrote transformed pcd to file " << pcd_file << std::endl;

  std::string trans_label_dir = out_label_dir + "/" + Name();
  if (!FileUtil::Exists(trans_label_dir)) {
    FileUtil::Mkdir(trans_label_dir);
  }
  std::string label_file = trans_label_dir + "/" + example.ExampleID() + ".label";
  example.GetLabel().ToFile(label_file);
  std::cout << "Wrote transformed label to file " << label_file << std::endl;
}
