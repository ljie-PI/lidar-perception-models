#pragma once

#include "common/example.h"
#include "transform.h"

class RotateZTransform : public Transform {
public:
  RotateZTransform(double ratio, double min_rot_angle, double max_rot_angle);

  ~RotateZTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;

private:
  std::shared_ptr<UniformDistRandom> ang_rand_ptr_;
};
