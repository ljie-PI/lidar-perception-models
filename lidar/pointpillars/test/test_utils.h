#pragma once

#include <cmath>
#include <fstream>
#include <string>

#include "common/string_util.h"
#include "common/pcl_defs.h"

bool LoadPointCloud(const std::string& pc_file,
                    LidarPointCloud *point_cloud) {
  std::ifstream pc_ifs(pc_file);
  if (!pc_ifs.is_open()) {
    return false;
  }
  std::string line;
  while (std::getline(pc_ifs, line)) {
    std::vector<std::string> tokens;
    StringUtil::split(line, &tokens);
    if (tokens.size() != 4) {
      continue;
    }
    LidarPoint point;
    point.x = std::stof(tokens[0]);
    point.y = std::stof(tokens[1]);
    point.z = std::stof(tokens[2]);
    point.intensity = std::stof(tokens[3]);
    point_cloud->push_back(point);
  }
  pc_ifs.close();
  return true;
}


int Offset2D(int x, int y, int x_size, int y_size) {
  return x * y_size + y;
}

bool DoubleNear(double d1, double d2, double error=1e-5) {
  double diff = d1 - d2;
  return diff > -std::fabs(error) && diff < std::fabs(error);
}