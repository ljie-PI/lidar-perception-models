#include "preprocessor.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>

#include "common/pcl_defs.h"
#include "common/protobuf_util.h"

using std::chrono::system_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

PreProcessor::PreProcessor(pointpillars::PointPillarsConfig& config)
    : config_(config) {
  voxel_generator_ = std::make_shared<VoxelGenerator>(config_);
  target_assigner_ = std::make_shared<TargetAssigner>(config_);
}

bool PreProcessor::SaveExample(const pointpillars::Example& example,
                               const std::string& example_id,
                               const std::string& output_dir) {
  std::string output_file = output_dir + "/" + example_id + ".bin";
  return ProtobufUtil::SaveToBinaryFile(example, output_file);
}

bool PreProcessor::SaveAnchors(const pointpillars::Example& example,
                               const std::string& example_id,
                               const std::string& output_dir) {
  std::string pos_anch_file = output_dir + "/pos_anchors/" + example_id + ".label";
  std::ofstream pos_anch_ofs(pos_anch_file);
  if (!pos_anch_ofs.is_open()) {
    std::cerr << "Failed to open " << pos_anch_file << " for write" << std::endl;
    return false;
  }
  std::string neg_anch_file = output_dir + "/neg_anchors/" + example_id + ".label";
  std::ofstream neg_anch_ofs(neg_anch_file);
  if (!neg_anch_ofs.is_open()) {
    std::cerr << "Failed to open " << neg_anch_file << " for write" << std::endl;
    return false;
  }

  for (int anch_id = 0; anch_id < example.anchor_size(); ++anch_id) {
    auto& anchor = example.anchor(anch_id);
    std::stringstream anchor_fmt_ss;
    anchor_fmt_ss << anchor.center_x() << " "
                  << anchor.center_y() << " "
                  << anchor.center_z() << " "
                  << anchor.length() << " "
                  << anchor.width() << " "
                  << anchor.height() << " "
                  << anchor.rotation() << " ";
    int label_id = anchor.target_label();
    if (label_id < 0) {
      anchor_fmt_ss << 0 << "\n";
    } else {
      anchor_fmt_ss << example.label(label_id).type() << "\n";
    }

    if (anchor.is_positive()) {
      pos_anch_ofs << anchor_fmt_ss.str();
    } else {
      neg_anch_ofs << anchor_fmt_ss.str();
    }
  }
  pos_anch_ofs.flush();
  pos_anch_ofs.close();
  neg_anch_ofs.flush();
  neg_anch_ofs.close();

  return true;
}

bool PreProcessor::Process(const std::string& example_id, const std::string& pcd_file,
                           const std::string& label_file, const std::string& output_dir,
                           const bool output_anchor) {

  LidarPointCloud point_cloud;
  // auto pcl_load_start = system_clock::now();
  if (pcl::io::loadPCDFile(pcd_file, point_cloud) < 0) {
    std::cerr << "Failed to load pcd file: " << pcd_file << std::endl;
    return false;
  }
  // auto pcl_load_end = system_clock::now();
  // std::cout << "Loading pcd cost "
  //           << duration_cast<milliseconds>(pcl_load_end - pcl_load_start).count()
  //           << " ms" << std::endl;
  std::vector<int> index_mapping;
  pcl::removeNaNFromPointCloud(point_cloud, point_cloud, index_mapping);
  Label label;
  if (!label.FromFile(label_file)) {
    std::cerr << "Failed to load label file: " << label_file << std::endl;
    return false;
  }

  pointpillars::Example example;
  // auto voxel_gen_start = system_clock::now();
  if (!voxel_generator_->Generate(point_cloud, &example)) {
    std::cerr << "Failed to generate voxels for example: " << example_id << std::endl;
    return false;
  }
  // auto voxel_gen_end = system_clock::now();
  // std::cout << "Generating voxel cost "
  //           << duration_cast<milliseconds>(voxel_gen_end - voxel_gen_start).count()
  //           << " ms" << std::endl;

  // auto target_assign_start = system_clock::now();
  if (!target_assigner_->Assign(point_cloud, label, &example)) {
    std::cerr << "Failed to generate labels for example: " << example_id << std::endl;
    return false;
  }
  // auto target_assign_end = system_clock::now();
  // std::cout << "Assigning targets cost "
  //           << duration_cast<milliseconds>(target_assign_end - target_assign_start).count()
  //           << " ms" << std::endl;

  // auto save_example_start = system_clock::now();
  if (!SaveExample(example, example_id, output_dir)) {
    std::cerr << "Failed to save example: " << example_id << std::endl;
    return false;
  }
  // auto save_example_end = system_clock::now();
  // std::cout << "Saveing example cost "
  //           << duration_cast<milliseconds>(save_example_end - save_example_start).count()
  //           << " ms" << std::endl;

  // auto save_anchor_start = system_clock::now();
  if (output_anchor) {
    SaveAnchors(example, example_id, output_dir);
  }
  // auto save_anchor_end = system_clock::now();
  // std::cout << "Saveing anchor cost "
  //           << duration_cast<milliseconds>(save_anchor_end - save_anchor_start).count()
  //           << " ms" << std::endl;


  return true;
}
