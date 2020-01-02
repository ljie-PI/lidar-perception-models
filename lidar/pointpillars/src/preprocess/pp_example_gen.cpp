#include <iostream>

#include <google/protobuf/text_format.h>

#include "common/file_util.h"
#include "common/string_util.h"
#include "common/protobuf_util.h"
#include "preprocessor.h"
#include "flags.h"

std::string LabelFileName(const std::string &example_id, const std::string &label_dir) {
  std::string label_file = example_id + ".label";
  if (label_dir.back() == '/') {
    return label_dir + label_file;
  }
  return label_dir + "/" + label_file;
}

bool process(const std::string& example,
             const std::string& pcd_file,
             const std::string& label_file) {
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  pointpillars::PointPillarsConfig pp_config;
  ProtobufUtil::ParseFromASCIIFile(FLAGS_config_file, &pp_config);

  PreProcessor preprocessor(pp_config);

  std::vector<std::string> pcd_files;
  FileUtil::GetFileList(FLAGS_input_pcd_dir, ".pcd", &pcd_files);
  for (auto &pcd_file : pcd_files) {
    int fname_start = static_cast<int>(pcd_file.rfind('/') + 1);
    std::string example_id = pcd_file.substr(fname_start, pcd_file.length() - fname_start - 4);
    std::string label_file = std::move(LabelFileName(example_id, FLAGS_input_label_dir));
    if (!FileUtil::Exists(label_file)) {
      std::cerr << "Label file(" << label_file << ") does not exist" << std::endl;
      return -1;
    }
    if (!preprocessor.Process(example_id, pcd_file, label_file, FLAGS_output_dir)) {
      std::cerr << "Failed to process example: " << example_id << std::endl;
      continue;
    }
  }

  return -1;
}