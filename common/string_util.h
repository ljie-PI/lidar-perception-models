#pragma once

#include <string>
#include <vector>
#include <sstream>

class StringUtil {
public:
  static int split(const std::string& s, std::vector<std::string> *vec, char ch=' ') {
    std::stringstream ss(s);
    std::string segment;
    int count = 0;
    while (std::getline(ss, segment, ch)) {
      vec->push_back(segment);
      ++count;
    }
    return count;
  }
};
