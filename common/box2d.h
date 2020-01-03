#pragma once

#include <cmath>
#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <Eigen/Core>

using Point2D = boost::geometry::model::d2::point_xy<float>;
using Polygon = boost::geometry::model::polygon<Point2D>;
using Corners = Eigen::Matrix<float, 2, 4>;
using boost::geometry::intersection;
using boost::geometry::union_;
using boost::geometry::area;
using boost::geometry::append;
using Eigen::Matrix2f;

class Box2D {
 public:
  Box2D(float x, float y, float length, float width, float yaw)
      : center_x_(x), center_y_(y),
        length_(length), width_(width), yaw_(yaw) {
    Matrix2f rot_mat;
    rot_mat <<
        std::cos(yaw), -std::sin(yaw),
        std::sin(yaw), std::cos(yaw);

    static int count = 0;
    Corners corners;
    corners <<
        -length * 0.5, length * 0.5, -length * 0.5, length * 0.5,
        -width * 0.5, -width * 0.5, width * 0.5, width * 0.5;
    Corners rot_corners = rot_mat * corners;
    for (int i = 0; i < 4; ++i) {
      rot_corners(0, i) = rot_corners(0, i) + x;
      rot_corners(1, i) = rot_corners(1, i) + y;
    }

    append(polygon_, Point2D(rot_corners(0, 0), rot_corners(1, 0)));
    append(polygon_, Point2D(rot_corners(0, 1), rot_corners(1, 1)));
    append(polygon_, Point2D(rot_corners(0, 2), rot_corners(1, 2)));
    append(polygon_, Point2D(rot_corners(0, 3), rot_corners(1, 3)));
    append(polygon_, Point2D(rot_corners(0, 0), rot_corners(1, 0)));

    min_x_ = max_x_ = rot_corners(0, 0);
    min_y_ = max_y_ = rot_corners(1, 0);
    for (int i = 1; i < 4; ++i) {
      min_x_ = std::min(min_x_, rot_corners(0, i));
      max_x_ = std::max(max_x_, rot_corners(0, i));
      min_y_ = std::min(min_y_, rot_corners(1, i));
      max_y_ = std::max(max_y_, rot_corners(1, i));
    }
  }

  double IouWith(const Box2D& other, int32_t criterion = -1) {
    std::vector<Polygon> overlap;
    intersection(polygon_, other.polygon_, overlap);

    double inter_area = overlap.empty() ? 0 : area(overlap.front());
    if (inter_area == 0.0) {
      return 0.0;
    }

    double union_area = area(polygon_) + area(other.polygon_) - inter_area;
    if (union_area == 0.0) {
      return 1.0;
    }

    return inter_area / union_area;
  }

  float MinX() { return min_x_; }
  float MaxX() { return max_x_; }
  float MinY() { return min_y_; }
  float MaxY() { return max_y_; }

 private:
  float center_x_;
  float center_y_;
  float length_;
  float width_;
  float yaw_;
  Polygon polygon_;
  float min_x_;
  float max_x_;
  float min_y_;
  float max_y_;
};

