#pragma once

#include <string>
#include <vector>
#include <sstream>

class StringUtil {
public:
  static void split(const std::string &s, std::vector<std::string> *vec) {
    std::istringstream ss(s);
    std::string tok;
    while (ss >> tok) {
      vec->push_back(tok);
    }
  }
};
