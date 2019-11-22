#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "common/random.h"
#include "common/example.h"

class Transform {
public:
  Transform(double ratio);

  ~Transform() = default;

  virtual bool Apply(const Example &ori_example,
                     Example *trans_example) = 0;

  std::string Name();

  bool ApplyByRatio(const Example &ori_example,
                    Example *trans_example);

  bool Save(const Example &example,
            const std::string &out_pcd_dir,
            const std::string &out_label_dir);

protected:
  double trans_ratio_;
  std::shared_ptr<UniformDistRandom> random_ptr_;
  std::string name_;
};