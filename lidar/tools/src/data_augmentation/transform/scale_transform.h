#pragma once

#include "transform.h"

class ScaleTransform : public Transform {
public:
  ScaleTransform(double ratio, double min_scale, double max_scale);

  ~ScaleTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;

private:
  std::shared_ptr<UniformDistRandom> factor_rand_ptr_;
};