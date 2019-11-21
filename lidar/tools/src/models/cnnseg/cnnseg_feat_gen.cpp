#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <gflags/gflags.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "common/file_util.h"

static bool ValidateStringNotEmpty(const char *flagname, const std::string &value) {
  return !value.empty();
}

DEFINE_string(input_pcd_dir, "", "directory of input origin pcd files");
DEFINE_validator(input_pcd_dir, &ValidateStringNotEmpty);
DEFINE_string(output_dir, "", "directory of output features");
DEFINE_validator(output_dir, &ValidateStringNotEmpty);

DEFINE_int32(height, 480, "height of 2d grids");
DEFINE_int32(width, 480, "width of 2d grids");
DEFINE_double(range, 40, "range in which obstacles are considered");
DEFINE_double(min_height, -5.0, "minimum height of LiDAR points");
DEFINE_double(max_height, 5.0, "maximum height of LiDAR points");

struct CNNSegFeature {
  int row;
  int col;
  float max_height;
  float mean_height;
  float log_count;
  float direction;
  float top_intensity;
  float mean_intensity;
  float distance;
  int nonempty;
};

static bool generate_features(const pcl::PointCloud<pcl::PointXYZI> &pc,
                              std::vector<std::shared_ptr<CNNSegFeature>> *features) {

}

static bool save_features(const std::vector<std::shared_ptr<CNNSegFeature>> &features,
                          const std::string filepath) {
  std::ofstream ofs(filepath);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open file " << filepath << " for write" << std::endl;
    return false;
  }
  for (auto &feature : features) {
    if (feature->nonempty == 1) {
      ofs << feature->row << ' '
          << feature->col << ' '
          << feature->max_height << ' '
          << feature->mean_height << ' '
          << feature->log_count << ' '
          << feature->direction << ' '
          << feature->top_intensity << ' '
          << feature->mean_intensity << ' '
          << feature->distance << ' '
          << feature->nonempty << '\n';
    }
  }
  ofs.flush();
  ofs.close();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!FileUtil::Exists(FLAGS_input_pcd_dir)) {
    std::cerr << "Input pcd directory " << FLAGS_input_pcd_dir << " doesn't exist" << std::endl;
    return -1;
  }
  if (!FileUtil::Exists(FLAGS_output_dir)) {
    std::cout << "Output directory " << FLAGS_output_dir << " doesn't exist, will create it" << std::endl;
    if (!FileUtil::Mkdir(FLAGS_output_dir)) {
      std::cerr << "Failed to create directory " << FLAGS_output_dir << std::endl;
    }
  }

  std::vector<std::string> pcd_files;
  FileUtil::GetFileList(FLAGS_input_pcd_dir, ".pcd", &pcd_files);

  for (auto &pcd_file : pcd_files) {
    int fname_start = pcd_file.rfind('/') + 1;
    std::string example_id = pcd_file.substr(fname_start, pcd_file.length() - fname_start - 4);

    pcl::PointCloud<pcl::PointXYZI> pc;
    if (pcl::io::loadPCDFile(pcd_file, pc) < 0) {
      std::cerr << "Failed to load pcd file: " << pcd_file << std::endl;
      continue;
    }

    std::vector<std::shared_ptr<CNNSegFeature>> features;
    if (!generate_features(pc, &features)) {
      std::cerr << "Failed to generate features for example: " << example_id << std::endl;
      continue;
    }

    std::string feat_file = FLAGS_output_dir + "/" + example_id + ".txt";
    if (!save_features(features, feat_file)) {
      std::cerr << "Failed to save features for example: " << example_id << std::endl;
    }
  }
  return 0;
}
