#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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
DEFINE_int32(channel_num, 8, "channel number of feature map");

DEFINE_int32(thread_num, 8, "number of threads");

static const double PI = 3.1415926535897932384626433832795;
static float *log_table = nullptr;
static const int MaxLogNum = 256;

static void InitLogTable() {
  log_table = new float[MaxLogNum];
  for (size_t i = 0; i < MaxLogNum; ++i) {
    log_table[i] = std::log(static_cast<float>(1 + i));
  }
}

float LogCount(int count) {
  if (count < static_cast<int>(MaxLogNum)) {
    return log_table[count];
  }
  return logf(static_cast<float>(1 + count));
}

static inline int offset(const int c = 0, const int h = 0, const int w = 0) {
  return (c * FLAGS_height + h) * FLAGS_width + w;
}

inline float Pixel2Pc(int in_pixel, float in_size, float out_range) {
  float res = 2.0f * out_range / in_size;
  return out_range - (static_cast<float>(in_pixel) + 0.5f) * res;
}

inline void GroupPc2Pixel(float pc_x, float pc_y, float scale, float range,
                          int* x, int* y) {
  float fx = (range - (0.707107f * (pc_x + pc_y))) * scale;
  float fy = (range - (0.707107f * (pc_x - pc_y))) * scale;
  *x = fx < 0 ? -1 : static_cast<int>(fx);
  *y = fy < 0 ? -1 : static_cast<int>(fy);
}

// map point to grid_id
inline int PointToGridID(float pt_x, float pt_y, float pt_z,
                         int height, int width, float range,
                         int min_height, int max_height, float inv_res_x = 0.0) {
  if (inv_res_x == 0.0) {
    inv_res_x = 0.5f * static_cast<float>(width) / range;
  }

  if (pt_z <= min_height || pt_z >= max_height) {
    return -1;
  }
  int pos_x = -1;
  int pos_y = -1;
  GroupPc2Pixel(pt_x, pt_y, inv_res_x, range, &pos_x, &pos_y);
  if (pos_y < 0 || pos_y >= height || pos_x < 0 || pos_x >= width) {
    return -1;
  }
  return pos_y * width + pos_x;
}

static bool GenerateFeatures(const pcl::PointCloud<pcl::PointXYZI> &pc, float *feat_data) {
  int channel_index = 0;
  int map_size = FLAGS_height * FLAGS_width;
  // access for convenience
  float *max_height_data = feat_data + offset(channel_index++);
  std::fill_n(max_height_data, map_size, FLAGS_min_height);
  float *mean_height_data = feat_data + offset(channel_index++);
  float *count_data = feat_data + offset(channel_index++);
  float *direction_data = feat_data + offset(channel_index++);
  float *top_intensity_data = feat_data + offset(channel_index++);
  float *mean_intensity_data = feat_data + offset(channel_index++);
  float *distance_data = feat_data + offset(channel_index++);
  float *nonempty_data = feat_data + offset(channel_index++);

  // extract constant features which are only affected by row and column
  for (int row = 0; row < FLAGS_height; ++row) {
    for (int col = 0; col < FLAGS_width; ++col) {
      int idx = row * FLAGS_width + col;
      float center_x = Pixel2Pc(row, static_cast<float>(FLAGS_height), FLAGS_range);
      float center_y = Pixel2Pc(col, static_cast<float>(FLAGS_width), FLAGS_range);
      direction_data[idx] = static_cast<float>(std::atan2(center_y, center_x) / (2.0 * PI));
      distance_data[idx] = static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
    }
  }

  // iterate point cloud to extract dynamic features
  float inv_res_x = 0.5f * static_cast<float>(FLAGS_width) / FLAGS_range;
  for (size_t i = 0; i < pc.size(); ++i) {
    const pcl::PointXYZI &pt = pc[i];
    int idx = PointToGridID(pt.x, pt.y, pt.z, FLAGS_height, FLAGS_width, FLAGS_range,
                            FLAGS_min_height, FLAGS_max_height, inv_res_x);
    if (idx == -1) {
      continue;
    }
    float pz = pt.z;
    float pi = float(pt.intensity) / 255.0f;
    if (max_height_data[idx] < pz) {
      max_height_data[idx] = pz;
      top_intensity_data[idx] = pi;
    }
    mean_height_data[idx] += static_cast<float>(pz);
    mean_intensity_data[idx] += static_cast<float>(pi);
    count_data[idx] += 1.f;
  }

  for (int i = 0; i < map_size; ++i) {
    if (count_data[i] <= FLT_EPSILON) {
      max_height_data[i] = 0.f;
    } else {
      mean_height_data[i] /= count_data[i];
      mean_intensity_data[i] /= count_data[i];
      nonempty_data[i] = 1.f;
    }
    count_data[i] = LogCount(static_cast<int>(count_data[i]));
  }
  return true;
}

static bool SaveFeatures(const float *feat_data, const std::string &filepath) {
  std::ofstream ofs(filepath);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open file " << filepath << " for write" << std::endl;
    return false;
  }
  int nonempty_chan = FLAGS_channel_num - 1;
  for (int row = 0; row < FLAGS_height; ++row) {
    for (int col = 0; col < FLAGS_width; ++ col) {
      if (feat_data[offset(nonempty_chan, row, col)] == 1.f) {
        ofs << row << ' ' << col << ' ';
        for (int chan = 0; chan < FLAGS_channel_num - 1; ++chan) {
          ofs << feat_data[offset(chan, row, col)] << ' ';
        }
        ofs << "1\n";
      }
    }
  }
  ofs.flush();
  ofs.close();
  return true;
}

void process(int init_idx, std::vector<std::string> *pcd_files) {
  int feat_data_size = FLAGS_channel_num * FLAGS_height * FLAGS_width;
  float *feat_data = new float[feat_data_size];
  int total_len = pcd_files->size();
  for (int i = init_idx; i < total_len; i+=FLAGS_thread_num) {
    std::string &pcd_file = pcd_files->at(i);
    int fname_start = pcd_file.rfind('/') + 1;
    std::string example_id = pcd_file.substr(fname_start, pcd_file.length() - fname_start - 4);

    pcl::PointCloud<pcl::PointXYZI> pc;
    auto start1 = std::chrono::system_clock::now();
    if (pcl::io::loadPCDFile(pcd_file, pc) < 0) {
      std::cerr << "Failed to load pcd file: " << pcd_file << std::endl;
      continue;
    }
    auto end1 = std::chrono::system_clock::now();
    std::cout << "load time cost: " << (end1 - start1).count()/1000000 << std::endl;

    auto start2 = std::chrono::system_clock::now();
    std::fill_n(feat_data, feat_data_size, 0.0);
    if (!GenerateFeatures(pc, feat_data)) {
      std::cerr << "Failed to generate features for example: " << example_id << std::endl;
      continue;
    }
    auto end2 = std::chrono::system_clock::now();
    std::cout << "extract time cost: " << (end2 - start2).count()/1000000 << std::endl;

    auto start3 = std::chrono::system_clock::now();
    std::string feat_file = FLAGS_output_dir + "/" + example_id + ".txt";
    if (!SaveFeatures(feat_data, feat_file)) {
      std::cerr << "Failed to save features for example: " << example_id << std::endl;
    }
    auto end3 = std::chrono::system_clock::now();
    std::cout << "save time cost: " << (end3 - start3).count()/1000000 << std::endl;
  }
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

  InitLogTable();

  std::cout << "Thread Number: " << FLAGS_thread_num << std::endl;
  std::vector<std::shared_ptr<std::thread>> tasks;

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    std::shared_ptr<std::thread> task = std::make_shared<std::thread>(process, tid, &pcd_files);
    tasks.push_back(task);
  }
  for (auto task : tasks) {
    task->join();
  }

  return 0;
}
