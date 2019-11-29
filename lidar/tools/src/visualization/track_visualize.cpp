#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <chrono>
#include <map>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "common/string_util.h"
#include "common/label.h"
#include "common/random.h"
#include "common/file_util.h"
#include "common/string_util.h"
#include "track_viz_flags.h"

typedef std::shared_ptr<Eigen::Vector3d> CenterPtr;
class HistoricalCenters {
public:
  HistoricalCenters() = default;
  ~HistoricalCenters() = default;

  void AddCenter(double timestamp, double x, double y, double z) {
    CenterPtr center = std::make_shared<Eigen::Vector3d>(x, y, z);
    hist_centers_.emplace(std::make_pair(timestamp, center));
  }

  std::map<double, CenterPtr>& GetHistCenters() {
    return hist_centers_;  
  }
private:  
  std::map<double, CenterPtr> hist_centers_;
};

struct Color {
  double r;
  double g;
  double b;
};

class ColorManager {
public:
  ColorManager() {
    red_rand_ = std::make_shared<UniformDistRandom>(0, 1.0);
    green_rand_ = std::make_shared<UniformDistRandom>(0, 1.0);
    blue_rand_ = std::make_shared<UniformDistRandom>(0, 1.0);
  }
  ~ColorManager() = default;

  std::shared_ptr<Color> GetColor(int id) {
    auto iter = color_map_.find(id);
    if (iter != color_map_.end()) {
      return iter->second;
    }
    std::shared_ptr<Color> new_color = RandomGenerate();
    color_map_.emplace(std::make_pair(id, new_color));
    return new_color;
  }
private:
  std::shared_ptr<Color> RandomGenerate() {
    std::shared_ptr<Color> color(new Color);
    color->r = red_rand_->Generate();
    color->g = green_rand_->Generate();
    color->b = blue_rand_->Generate();
    return color;
  }
  std::map<int, std::shared_ptr<Color>> color_map_;
  std::shared_ptr<UniformDistRandom> red_rand_;
  std::shared_ptr<UniformDistRandom> green_rand_;
  std::shared_ptr<UniformDistRandom> blue_rand_;
};

class TrackVisualizerWrapper {
public:
  TrackVisualizerWrapper() = default;
  ~TrackVisualizerWrapper() = default;

  pcl::visualization::PCLVisualizer &GetVisualizer() {
    return pcl_visualizer_;
  }

  void AddFramePose(const std::string& frame_id, std::shared_ptr<Eigen::Affine3d> pose_ptr) {
    frame_pose_map_.emplace(frame_id, pose_ptr);
  }

  std::shared_ptr<Eigen::Affine3d> GetFramePose(const std::string& frame_id) {
    auto iter = frame_pose_map_.find(frame_id);
    if (iter != frame_pose_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

  void AddFrameTime(const std::string& frame_id, double timestamp) {
    frame_time_map_.emplace(frame_id, timestamp);
  }

  double GetFrameTime(const std::string& frame_id) {
    auto iter = frame_time_map_.find(frame_id);
    if (iter != frame_time_map_.end()) {
      return iter->second;
    }
    return 0.0;
  }

  void AddHistoricalCenter(int track_id, double timestamp, double x, double y, double z) {
    std::shared_ptr<HistoricalCenters> hist_centers;
    auto track_iter = track_hist_centers_.find(track_id);
    if (track_iter == track_hist_centers_.end()) {
      hist_centers = std::make_shared<HistoricalCenters>();
      track_hist_centers_.emplace(
        std::make_pair(track_id, hist_centers));
    } else {
      hist_centers = track_iter->second;
    }
    hist_centers->AddCenter(timestamp, x, y, z);
  }

  std::map<int, std::shared_ptr<HistoricalCenters>> &GetTrackHistoricalCenters() {
    return track_hist_centers_;
  }

  ColorManager& TrackPointColorManager() {
    return track_point_cm_;
  }
private:
  pcl::visualization::PCLVisualizer pcl_visualizer_;
  ColorManager track_point_cm_;
  std::map<int, std::shared_ptr<HistoricalCenters>> track_hist_centers_;
  std::map<std::string, std::shared_ptr<Eigen::Affine3d>> frame_pose_map_;
  std::map<std::string, double> frame_time_map_;
};

bool LoadPose(const std::string &pose_file, Eigen::Affine3d *pose, double *timestamp) {
  std::ifstream fin(pose_file.c_str());
  if (!fin.is_open()) {
    std::cerr << "Failed to open pose file: " << pose_file << std::endl;
    return false;
  }

  Eigen::Vector3d translation;
  Eigen::Quaterniond quat;
  long frame_id;
  fin >> frame_id >> *timestamp >> translation(0) >> translation(1) >>
      translation(2) >> quat.x() >> quat.y() >> quat.z() >> quat.w();
  *pose = Eigen::Affine3d::Identity();
  pose->prerotate(quat);
  pose->pretranslate(translation);
  fin.close();
  return true;
}

void LoadOneBBox(std::vector<std::string> &tokens, TrackVisualizerWrapper &vis_wrapper,
                 Eigen::Affine3d& pose_inv, std::vector<BoundingBox> *bboxes) {
  double timestamp = std::stod(tokens[0]);
  int track_id = std::stoi(tokens[1]);
  double center_x = std::stod(tokens[2]);
  double center_y = std::stod(tokens[3]);
  double center_z = std::stod(tokens[4]);
  vis_wrapper.AddHistoricalCenter(track_id, timestamp, center_x, center_y, center_z);
  double theta = std::stod(tokens[8]);
  Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();
  trans_mat(0, 0) = std::cos(theta);
  trans_mat(0, 1) = -std::sin(theta);
  trans_mat(1, 0) = std::sin(theta);
  trans_mat(1, 1) = std::cos(theta);
  trans_mat(0, 3) = center_x;
  trans_mat(1, 3) = center_y;
  trans_mat(2, 3) = center_z;
  trans_mat(3, 3) = 1;
  trans_mat = pose_inv.matrix() * trans_mat;

  Eigen::Vector3d direction(std::cos(theta), std::sin(theta), 0);
  direction = pose_inv.rotation() * direction;
  theta = std::atan2(direction.y(), direction.x());
  BoundingBox bbox {
    trans_mat(0, 3),
    trans_mat(1, 3),
    trans_mat(2, 3),
    std::stod(tokens[5]),
    std::stod(tokens[6]),
    std::stod(tokens[7]),
    theta,
    std::stoi(tokens[9])
  };
  bboxes->emplace_back(bbox);
}

void LoadBBoxes(const std::string &frame_id,
                TrackVisualizerWrapper &vis_wrapper,
                std::vector<BoundingBox> *bboxes) {
  std::vector<std::string> lines;
  std::string track_path = FLAGS_input_track_dir + "/" + frame_id + ".track";
  if (!FileUtil::ReadLines(track_path, &lines)) {
    std::cerr << "Failed to load bounding-boxes from: " << track_path << std::endl;
    return;
  }
  std::shared_ptr<Eigen::Affine3d> pose = vis_wrapper.GetFramePose(frame_id);
  std::cout << "pose of frame: " << frame_id << "\n";
  std::cout << std::to_string(pose->translation().x()) << ", " << std::to_string(pose->translation().y()) << ", " << pose->translation().z() << "\n";
  std::cout << pose->inverse().translation().x() << ", " << std::to_string(pose->inverse().translation().y()) << ", " << pose->inverse().translation().z() << "\n";
  std::cout << std::endl;

  Eigen::Affine3d pose_inv = pose->inverse();
  int last_track_id = INT_MIN;
  double max_ts = DBL_MIN;
  std::vector<std::string> max_ts_tokens;
  std::vector<std::string> tokens;
  for (const auto &line : lines) {
    tokens.clear();
    StringUtil::split(line, &tokens);
    if (tokens.size() != 10) {
      continue;
    }
    double timestamp = std::stol(tokens[0]);
    int track_id = std::stoi(tokens[1]);
    if (track_id != last_track_id && last_track_id != INT_MIN) {
      LoadOneBBox(max_ts_tokens, vis_wrapper, pose_inv, bboxes);
      max_ts_tokens = tokens;
    } else {
      if (timestamp > max_ts) {
        max_ts = timestamp;
        max_ts_tokens = tokens;
      }
    }
    last_track_id = track_id;
  }
  // load last bounding-box
  LoadOneBBox(max_ts_tokens, vis_wrapper, pose_inv, bboxes);
}

void AddCurrentFramePoints(
    const std::string &pcd_path, pcl::visualization::PCLVisualizer &visualizer) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  if (pcl::io::loadPCDFile(pcd_path, *pc_ptr) < 0) {
    std::cerr << "Failed to load pcd file: " << pcd_path << std::endl;
    return;
  }
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_handler(pc_ptr, 255, 255, 255);
  std::string pc_id = "point_cloud";
  visualizer.addPointCloud(pc_ptr, color_handler, pc_id);
  visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, pc_id);
}

void AddCurrentFrameBBoxes(std::vector<BoundingBox> &bboxes,
                           pcl::visualization::PCLVisualizer &visualizer) {
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const BoundingBox &bbox = bboxes[i];
    std::string cube_id = "cube_" + std::to_string(i);
    Eigen::Vector3f center(bbox.center_x, bbox.center_y, bbox.center_z);
    Eigen::Quaternionf rotation(std::cos(bbox.heading / 2), 0, 0, std::sin(bbox.heading / 2));
    visualizer.addCube(center, rotation, bbox.length, bbox.width, bbox.height, cube_id);
    visualizer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                           pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube_id);
    visualizer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                           0.8, 0.2, 0.2, cube_id);

    Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
    trans_mat(0, 0) = std::cos(bbox.heading);
    trans_mat(0, 1) = -std::sin(bbox.heading);
    trans_mat(1, 0) = std::sin(bbox.heading);
    trans_mat(1, 1) = std::cos(bbox.heading);
    trans_mat(0, 3) = bbox.center_x;
    trans_mat(1, 3) = bbox.center_y;
    trans_mat(2, 3) = bbox.center_z;

    Eigen::Vector4f arrow_start(bbox.length, 0, 0, 1);
    arrow_start = trans_mat * arrow_start;

    pcl::PointXYZ pt1(center.x(), center.y(), center.z());
    pcl::PointXYZ pt2(arrow_start.x(), arrow_start.y(), arrow_start.z());
    std::string line_id = "line_" + std::to_string(i);
    visualizer.addLine(pt1, pt2, 0.2, 0.8, 0.2, line_id);
    visualizer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, line_id);
  }
}

void AddHistoricalTracks(TrackVisualizerWrapper &vis_wrapper,
                         const std::string& frame_id) {
  std::map<int, std::shared_ptr<HistoricalCenters>> &track_hist_centers
      = vis_wrapper.GetTrackHistoricalCenters();
  double current_ts = vis_wrapper.GetFrameTime(frame_id);
  if (current_ts <= 0.0) {
    std::cerr << "Failed to get current timestamp" << std::endl;
    return;
  }
  double oldest_ts = current_ts - 10;
  std::shared_ptr<Eigen::Affine3d> current_pose = vis_wrapper.GetFramePose(frame_id);
  if (current_pose == nullptr) {
    std::cerr << "Failed to get current pose" << std::endl;
    return;
  }
  Eigen::Affine3d current_inv = current_pose->inverse();
  
  for (auto &hist_centers_pair : track_hist_centers) {
    int track_id = hist_centers_pair.first;
    auto &hist_centers = hist_centers_pair.second->GetHistCenters();
    for (auto iter = hist_centers.begin(); iter != hist_centers.end(); ++iter) {
      if (iter->first < oldest_ts) {
        hist_centers.erase(iter);
      } else {
        CenterPtr center = iter->second;
        Eigen::Vector3d trans_center = current_inv * (*center);
        pcl::PointXYZ center_point(trans_center.x(), trans_center.y(), trans_center.z());
        std::shared_ptr<Color> color = vis_wrapper.TrackPointColorManager().GetColor(track_id);
        std::string sphere_id = "sphere_" + std::to_string(track_id) + "_" + std::to_string(iter->first);
        vis_wrapper.GetVisualizer().addSphere(
          center_point, 0.2, color->r, color->g, color->b, sphere_id);
      }
    }
  }
}

void Visualize(TrackVisualizerWrapper &vis_wrapper,
               const std::string &id, double spin_times) {
  std::string pcd_path = FLAGS_input_pcd_dir + "/" + id + ".pcd";

  pcl::visualization::PCLVisualizer &visualizer = vis_wrapper.GetVisualizer();
  // reset when creating a visualizer window
  if (visualizer.wasStopped()) {
    visualizer.resetStoppedFlag();
  }
  visualizer.setBackgroundColor(0, 0, 0);

  // add points in current frame
  AddCurrentFramePoints(pcd_path, visualizer);

  // add boundingboxes in current frame
  std::vector<BoundingBox> bboxes;
  LoadBBoxes(id, vis_wrapper, &bboxes);
  AddCurrentFrameBBoxes(bboxes, visualizer);

  // add historical tracks
  AddHistoricalTracks(vis_wrapper, id);

  visualizer.addCoordinateSystem(5.0);
  visualizer.setCameraPosition(10, -10, 50, 0, 0, 0);
  long times = 0;
  while (!visualizer.wasStopped()) {
    visualizer.spinOnce(10);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    times += 1;
    if (spin_times > 0 && times > spin_times) {
      break;
    }
  }
  visualizer.removeAllCoordinateSystems();
  visualizer.removeAllShapes();
  visualizer.removeAllPointClouds();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> id_list;
  StringUtil::split(FLAGS_id_list, &id_list, ',');

  TrackVisualizerWrapper vis_wrapper;
  for (const auto &frame_id : id_list) {
    std::string pose_file = FLAGS_input_pose_dir + "/" + frame_id + ".pose";
    std::shared_ptr<Eigen::Affine3d> pose_ptr(new Eigen::Affine3d);
    double timestamp;
    if (!LoadPose(pose_file, pose_ptr.get(), &timestamp)) {
      std::cerr << "Failed to load pose form " << pose_file << std::endl;
      continue;
    }
    vis_wrapper.AddFrameTime(frame_id, timestamp);
    vis_wrapper.AddFramePose(frame_id, pose_ptr);
  }

  for (const auto &frame_id : id_list) {
    Visualize(vis_wrapper, frame_id, FLAGS_spin_times);
  }
}
