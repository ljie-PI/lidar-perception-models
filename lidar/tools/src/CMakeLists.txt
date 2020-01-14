cmake_minimum_required(VERSION 3.10)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lidar)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
file(GLOB COMMON_SRC_FILES ${COMMON_SRC_DIR}/*.cpp)

## data augmentation
set(DATA_AUG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_augmentation)
set(DATA_AUG_TARGET data_aug)
file(GLOB_RECURSE DATA_AUG_SRC_FILES ${DATA_AUG_SRC_DIR}/*.cpp)
message(STATUS "DATA_AUG_SRC_FILES: ${DATA_AUG_SRC_FILES}")
add_executable(${DATA_AUG_TARGET}
        ${COMMON_SRC_FILES}
        ${DATA_AUG_SRC_FILES})
target_link_libraries(${DATA_AUG_TARGET}
        gflags
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES})

## visualization
set(VIS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/visualization)
set(SEG_VIS_TARGET seg_view)
add_executable(${SEG_VIS_TARGET}
        ${COMMON_SRC_FILES}
        ${VIS_SRC_DIR}/seg_visualize.cpp)
target_link_libraries(${SEG_VIS_TARGET}
        gflags
        ${PCL_LIBRARIES})

set(TRACK_VIS_TARGET track_view)
add_executable(${TRACK_VIS_TARGET}
        ${COMMON_SRC_FILES}
        ${VIS_SRC_DIR}/track_visualize.cpp)
target_link_libraries(${TRACK_VIS_TARGET}
        gflags
        ${PCL_LIBRARIES})

## metric calculation
