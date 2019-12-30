#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

class FileUtil {
public:
  static bool GetFileList(const std::string &path,
                          const std::string &suffix,
                          std::vector<std::string> *files) {
    if (!boost::filesystem::exists(path)) {
      std::cerr << path << " does not exist." << std::endl;
      return false;
    }
    boost::filesystem::recursive_directory_iterator itr(path);
    while (itr != boost::filesystem::recursive_directory_iterator()) {
      try {
        const std::string &filename = itr->path().string();
        if (filename.rfind(suffix) == filename.length() - suffix.length()) {
          files->push_back(filename);
        }
        ++itr;
      } catch (const std::exception &ex) {
        std::cerr << "Caught execption: " << ex.what() << std::endl;
        continue;
      }
    }
    return true;
  }

  static bool Exists(const std::string &path) {
    return boost::filesystem::exists(path);
  }

  static bool ReadLines(const std::string &filepath, std::vector<std::string> *lines) {
    if (!FileUtil::Exists(filepath)) {
      return false;
    }
    std::ifstream ifs(filepath);
    std::string line;
    while (std::getline(ifs, line)) {
      lines->push_back(line);
    }
    ifs.close();
    return true;
  }

  static bool WriteLines(const std::vector<std::string> &lines, const std::string &filepath) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
      return false;
    }
    for (const auto &line : lines) {
      ofs << line << "\n";
    }
    ofs.flush();
    ofs.close();
  }

  static bool Mkdir(std::string dir) {
    return boost::filesystem::create_directories(dir);
  }
};
