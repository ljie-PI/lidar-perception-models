#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>

struct BoundingBox {
  double center_x;
  double center_y;
  double center_z;
  double length;
  double width;
  double height;
  double heading;
  int type;

  void GetCorners(std::vector<Eigen::Vector4f> *corners);
};

class Label {
public:
  Label() = default;

  Label(std::vector<BoundingBox> &bboxes);

  ~Label() = default;

  bool FromFile(const std::string &label_file);

  bool ToFile(const std::string &label_file) const;

  const std::vector<BoundingBox> &BoundingBoxes() const;

  void ModBoundingBox(int id, BoundingBox bbox);

  bool DeepCopy(const Label &other);

  int BoundingBoxCount() const;
  void SetBoundingBoxCount(int count);

private:
  BoundingBox ParseBox(const std::string &line);

  std::vector<BoundingBox> bboxes_;
  int box_cnt_;
};