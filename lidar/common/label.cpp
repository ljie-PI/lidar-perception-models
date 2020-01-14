#include <cmath>

#include "common/file_util.h"
#include "common/string_util.h"
#include "label.h"

void BoundingBox::GetCorners(std::vector<Eigen::Vector4f> *corners) {
  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
  trans_mat(0, 0) = std::cos(heading);
  trans_mat(0, 1) = -std::sin(heading);
  trans_mat(1, 0) = std::sin(heading);
  trans_mat(1, 1) = std::cos(heading);
  trans_mat(0, 3) = center_x;
  trans_mat(1, 3) = center_y;
  trans_mat(2, 3) = center_z;

  Eigen::Vector4f front_left_bottom(length/2, -width/2, -height/2, 1);
  front_left_bottom = trans_mat * front_left_bottom;
  corners->push_back(front_left_bottom);

  Eigen::Vector4f front_left_up(length/2, -width/2, height/2, 1);
  front_left_up = trans_mat * front_left_up;
  corners->push_back(front_left_up);

  Eigen::Vector4f front_right_bottom(length/2, width/2, -height/2, 1);
  front_right_bottom = trans_mat * front_right_bottom;
  corners->push_back(front_right_bottom);

  Eigen::Vector4f front_right_up(length/2, width/2, height/2, 1);
  front_right_up = trans_mat * front_right_up;
  corners->push_back(front_right_up);

  Eigen::Vector4f back_left_bottom(-length/2, -width/2, -height/2, 1);
  back_left_bottom = trans_mat * back_left_bottom;
  corners->push_back(back_left_bottom);

  Eigen::Vector4f back_left_up(-length/2, -width/2, height/2, 1);
  back_left_up = trans_mat * back_left_up;
  corners->push_back(back_left_up);

  Eigen::Vector4f back_right_bottom(-length/2, width/2, -height/2, 1);
  back_right_bottom = trans_mat * back_right_bottom;
  corners->push_back(back_right_bottom);

  Eigen::Vector4f back_right_up(-length/2, width/2, height/2, 1);
  back_right_up = trans_mat * back_left_up;
  corners->push_back(back_right_up);
}

Label::Label(std::vector<BoundingBox> &bboxes) {
  bboxes_ = bboxes;
  box_cnt_ = bboxes.size();
}

bool Label::FromFile(const std::string &label_file) {
  box_cnt_ = 0;
  std::vector<std::string> lines;
  if (!FileUtil::ReadLines(label_file, &lines)) {
    return false;
  }
  bboxes_.clear();
  for (const auto &line : lines) {
    bboxes_.push_back(std::move(ParseBox(line)));
    ++box_cnt_;
  }
  return true;
}

BoundingBox Label::ParseBox(const std::string &line) {
  std::vector<std::string> tokens;
  StringUtil::split(line, &tokens);
  BoundingBox bbox {
    std::stod(tokens[0]),
    std::stod(tokens[1]),
    std::stod(tokens[2]),
    std::stod(tokens[3]),
    std::stod(tokens[4]),
    std::stod(tokens[5]),
    std::stod(tokens[6]),
    std::stoi(tokens[7])
  };
  return bbox;
}

bool Label::ToFile(const std::string &label_file) const {
  std::vector<std::string> lines;
  for (int i = 0; i < box_cnt_; ++i) {
    BoundingBox bbox = bboxes_[i];
    lines.emplace_back(
      std::to_string(bbox.center_x) + " " +
      std::to_string(bbox.center_y) + " " +
      std::to_string(bbox.center_z) + " " +
      std::to_string(bbox.length) + " " +
      std::to_string(bbox.width) + " " +
      std::to_string(bbox.height) + " " +
      std::to_string(bbox.heading) + " " +
      std::to_string(bbox.type)
    );
  }
  FileUtil::WriteLines(lines, label_file);
  return false;
}

const std::vector<BoundingBox> &Label::BoundingBoxes() const {
  return bboxes_;
}

bool Label::DeepCopy(const Label &other){
  const std::vector<BoundingBox> &bboxes = other.BoundingBoxes();
  for (const auto &bbox : bboxes) {
    BoundingBox newBbox{
        bbox.center_x,
        bbox.center_y,
        bbox.center_z,
        bbox.length,
        bbox.width,
        bbox.height,
        bbox.heading,
        bbox.type
    };
    bboxes_.emplace_back(newBbox);
  }
  box_cnt_ = bboxes_.size();
}

void Label::ModBoundingBox(int id, BoundingBox bbox) {
  if (id >= bboxes_.size()) {
    return;
  }
  BoundingBox newBbox{
      bbox.center_x,
      bbox.center_y,
      bbox.center_z,
      bbox.length,
      bbox.width,
      bbox.height,
      bbox.heading,
      bbox.type
  };
  bboxes_[id] = newBbox;
}

int Label::BoundingBoxCount() const {
  return box_cnt_;
}

void Label::SetBoundingBoxCount(int count) {
  bboxes_.resize(count);
  box_cnt_ = count;
}
