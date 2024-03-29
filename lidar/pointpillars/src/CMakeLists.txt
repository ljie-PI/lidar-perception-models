cmake_minimum_required(VERSION 3.10)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Protobuf.cmake)

include_directories(${CMAKE_SOURCE_DIR}) 
include_directories(${CMAKE_SOURCE_DIR}/lidar)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../proto)

link_directories(${CMAKE_BINARY_DIR}/lib)

set(COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
file(GLOB COMMON_SRC_FILES ${COMMON_SRC_DIR}/*.cpp)

## generate pointpillars examples
set(PP_EXAMPLE_GEN pp_example_gen)
file(GLOB PP_EXAMPLE_GEN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/*.cpp)
add_executable(${PP_EXAMPLE_GEN}
        ${COMMON_SRC_FILES}
        ${PP_EXAMPLE_GEN_SRCS})
target_link_libraries(${PP_EXAMPLE_GEN}
        pp_proto_defs
        gflags
        ${Protobuf_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES})
add_dependencies(${PP_EXAMPLE_GEN} pp_proto_defs)
