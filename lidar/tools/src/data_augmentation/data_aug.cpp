#include <thread>
#include <pcl/visualization/pcl_visualizer.h>

#include "flags.h"
#include "common/file_util.h"
#include "common/label.h"
#include "common/example.h"
#include "transform/transform.h"
#include "transform/reflect_x_transform.h"
#include "transform/reflect_y_transform.h"
#include "transform/rotate_z_transform.h"
#include "transform/scale_transform.h"
#include "transform/move_transform.h"
#include "transform/down_sample_transform.h"
#include "transform/up_sample_transform.h"
#include "transform/ground_filter_transform.h"

std::string LabelFileName(const std::string &example_id, const std::string &label_dir) {
  std::string label_file = example_id + ".label";
  if (label_dir.back() == '/') {
    return label_dir + label_file;
  }
  return label_dir + "/" + label_file;
}

std::vector<std::shared_ptr<Transform>> GetTransforms() {
  std::vector<std::shared_ptr<Transform>> transforms;
  if (FLAGS_reflect_x) {
    std::shared_ptr<Transform> reflect_x_ptr = std::make_shared<ReflectXTransform>(FLAGS_reflect_x_ratio);
    transforms.push_back(reflect_x_ptr);
  }
  if (FLAGS_reflect_y) {
    std::shared_ptr<Transform> reflect_y_ptr = std::make_shared<ReflectYTransform>(FLAGS_reflect_y_ratio);
    transforms.push_back(reflect_y_ptr);
  }
  if (FLAGS_rotate_z) {
    std::shared_ptr<Transform> rotate_z_ptr =
        std::make_shared<RotateZTransform>(FLAGS_rotate_z_ratio, FLAGS_min_rot_angle, FLAGS_max_rot_angle);
    transforms.push_back(rotate_z_ptr);
  }
  if (FLAGS_scale) {
    std::shared_ptr<Transform> scale_ptr =
        std::make_shared<ScaleTransform>(FLAGS_scale_ratio, FLAGS_min_scale, FLAGS_max_scale);
    transforms.push_back(scale_ptr);
  }
  if (FLAGS_move) {
    std::shared_ptr<Transform> move_ptr =
        std::make_shared<MoveTransform>(FLAGS_move_ratio, FLAGS_move_mean, FLAGS_move_std);
    transforms.push_back(move_ptr);
  }
  if (FLAGS_down_sample) {
    std::shared_ptr<Transform> down_sample_ptr =
        std::make_shared<DownSampleTransform>(FLAGS_down_sample_ratio, FLAGS_down_sample_range, FLAGS_down_sample_factor);
    transforms.push_back(down_sample_ptr);
  }
  if (FLAGS_up_sample) {
    std::shared_ptr<Transform> up_sample_ptr =
        std::make_shared<UpSampleTransform>(FLAGS_up_sample_ratio, FLAGS_up_sample_range, FLAGS_up_sample_factor);
    transforms.push_back(up_sample_ptr);
  }
  if (FLAGS_ground_filter) {
    std::shared_ptr<Transform> ground_filter_ptr =
        std::make_shared<GroundFilterTransform>(FLAGS_ground_filter_ratio, FLAGS_ground_height);
    transforms.push_back(ground_filter_ptr);
  }
  return transforms;
}

void process(int init_idx, std::vector<std::string> *ori_pcd_files,
             std::vector<std::shared_ptr<Transform>> *transforms) {
  int total_len = ori_pcd_files->size();
  for (int i = init_idx; i < total_len; i+=FLAGS_thread_num) {
    std::string &ori_pcd_file = ori_pcd_files->at(i);
    int fname_start = ori_pcd_file.rfind('/') + 1;
    std::string example_id = ori_pcd_file.substr(fname_start, ori_pcd_file.length() - fname_start - 4);
    std::string label_file = std::move(LabelFileName(example_id, FLAGS_input_label_dir));
    if (FileUtil::Exists(label_file)) {
      Example ori_example(example_id, ori_pcd_file, label_file);
      if (ori_example.IsValid()) {
        std::cout << "====================" << std::endl;
        std::cout << "Processing example: " << ori_example.ExampleID() << std::endl;
        std::cout << "Label contains " << ori_example.GetLabel().BoundingBoxCount() << " obstacles" << std::endl;
        for (int trans_idx = 0; trans_idx < transforms->size(); ++trans_idx) {
          std::shared_ptr<Transform> transform = transforms->at(trans_idx);
          Example trans_example;
          transform->ApplyByRatio(ori_example, &trans_example);
          if (trans_example.IsValid()) {
            transform->Save(trans_example, FLAGS_output_pcd_dir, FLAGS_output_label_dir);
          }
        }
      }
    } else {
      std::cerr << label_file << " does not exist" << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::shared_ptr<Transform>> transforms = std::move(GetTransforms());
  std::cout << "Will using " << transforms.size() << " transforms: \n";
  for (const auto& transform : transforms) {
    std::cout << "\t" << transform->Name() << "\n";
  }
  std::cout << std::endl;

  std::vector<std::string> ori_pcd_files;
  FileUtil::GetFileList(FLAGS_input_pcd_dir, ".pcd", &ori_pcd_files);

  std::vector<std::shared_ptr<std::thread>> tasks;
  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    std::shared_ptr<std::thread> task = std::make_shared<std::thread>(process, tid, &ori_pcd_files, &transforms);
    tasks.push_back(task);
  }
  for (auto task : tasks) {
    task->join();
  }
  return 0;
}
