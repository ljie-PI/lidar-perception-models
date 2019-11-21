#pragma once

#include "common/example.h"
#include "transform.h"

class ReflectXTransform : public Transform {
public:
  ReflectXTransform(double ratio);

  ~ReflectXTransform() = default;

  bool Apply(const Example &ori_example, Example *trans_example) override ;

};
