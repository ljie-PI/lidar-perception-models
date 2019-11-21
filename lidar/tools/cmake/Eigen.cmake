find_package(Eigen3 REQUIRED eigen3)
set(USE_EIGEN 1)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARY_DIRS})