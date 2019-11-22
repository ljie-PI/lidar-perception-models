cmake_minimum_required(VERSION 2.8)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
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
        gtest
        gflags
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES})

## visualization
set(VIS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/visualization)
set(VIS_TARGET visualize)
add_executable(${VIS_TARGET}
        ${COMMON_SRC_FILES}
        ${VIS_SRC_DIR}/visualize.cpp)
target_link_libraries(${VIS_TARGET}
        gtest
        gflags
        ${PCL_LIBRARIES})

## metric calculation

## model specific
add_subdirectory(models/cnnseg)
