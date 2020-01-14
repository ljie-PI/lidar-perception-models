cmake_minimum_required(VERSION 3.10)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

## feature generate
set(CNNSEG_FEAT_GEN cnnseg_feat_gen)
add_executable(${CNNSEG_FEAT_GEN}
        ${COMMON_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/cnnseg_feat_gen.cpp)
target_link_libraries(${CNNSEG_FEAT_GEN}
        gtest
        gflags
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES})
