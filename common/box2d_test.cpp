#include "box2d.h"

#include <cmath>
#include <gtest/gtest.h>

TEST(Box2DTest, iou_test) {
  Box2D box1(0.0, 0.0, sqrt(2.0) * 2, sqrt(2.0), M_PI_4);
  Box2D box2(1.0, -1.0, 2.0, 2.0, 0);
  EXPECT_NEAR(0.5 / 7.5, box1.IouWith(box2), 1e-5);
  EXPECT_NEAR(0.5 / 7.5, box2.IouWith(box1), 1e-5);

  Box2D box3(0.0, 0.0, sqrt(2.0), sqrt(2.0), M_PI_4);
  Box2D box4(0.0, 0.0, 2.0, 2.0, 0);
  EXPECT_NEAR(0.5, box4.IouWith(box3), 1e-5);
  EXPECT_NEAR(0.5, box3.IouWith(box4), 1e-5);

  Box2D box5(5.0, 5.0, 2.0, 2.0, 0);
  EXPECT_NEAR(0.0, box1.IouWith(box5), 1e-5);
  EXPECT_NEAR(0.0, box5.IouWith(box1), 1e-5);
}