#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "common/string_util.h"
#include "common/label.h"
#include "seg_viz_flags.h"

void visualize(pcl::visualization::PCLVisualizer &visualizer,
               const std::string &id, double spin_times) {
  std::string pcd_path = FLAGS_input_pcd_dir + "/" + id + ".pcd";
  std::string label_path = FLAGS_input_label_dir + "/" + id + ".label";

  if (visualizer.wasStopped()) {
    visualizer.resetStoppedFlag();
  }
  visualizer.setBackgroundColor(0, 0, 0);

  // add points
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  if (pcl::io::loadPCDFile(pcd_path, *pc_ptr) < 0) {
    std::cerr << "Failed to load pcd file: " << pcd_path << std::endl;
    return;
  }
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_handler(pc_ptr, 255, 255, 255);
  std::string pc_id = "point_cloud";
  visualizer.addPointCloud(pc_ptr, color_handler, pc_id);
  visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, pc_id);

  // add labels
  Label label;
  if (!label.FromFile(label_path)) {
    std::cerr << "Failed to load label file: " << label_path << std::endl;
    return;
  }
  std::vector<BoundingBox> bboxes = label.BoundingBoxes();
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

  pcl::visualization::PCLVisualizer visualizer;
  for (const auto &id : id_list) {
    visualize(visualizer, id, FLAGS_spin_times);
  }
}
