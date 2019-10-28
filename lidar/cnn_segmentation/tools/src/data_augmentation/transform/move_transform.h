#pragma once

#include "transform.h"

class MoveTransform : public Transform {
public:
  MoveTransform(double ratio, double move_mean, double move_std);

  ~MoveTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;

private:
  std::shared_ptr<NormalDistRandom> x_rand_ptr_;
  std::shared_ptr<NormalDistRandom> y_rand_ptr_;
  std::shared_ptr<NormalDistRandom> z_rand_ptr_;
};