#pragma once

#include "transform.h"

class GroundFilterTransform : public Transform {
public:
  GroundFilterTransform(double ratio, double ground_height);

  ~GroundFilterTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;

private:
  double ground_height_;
};