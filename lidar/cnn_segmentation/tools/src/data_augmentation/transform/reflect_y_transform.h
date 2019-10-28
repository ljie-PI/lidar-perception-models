#pragma once

#include "common/example.h"
#include "transform.h"

class ReflectYTransform : public Transform {
public:
  ReflectYTransform(double ratio);

  ~ReflectYTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override;
};
