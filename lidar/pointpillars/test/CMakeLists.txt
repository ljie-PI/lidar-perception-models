cmake_minimum_required(VERSION 3.10)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Protobuf.cmake)

include_directories(${CMAKE_SOURCE_DIR}) 
include_directories(${CMAKE_SOURCE_DIR}/lidar)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../proto)

link_directories(${CMAKE_BINARY_DIR}/lib)

set(COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
file(GLOB COMMON_SRC_FILES ${COMMON_SRC_DIR}/*.cpp)
set(PREPROCESS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../src/preprocess/voxel_mapping.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/../src/preprocess/voxel_generator.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/../src/preprocess/target_assigner.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/../src/preprocess/preprocessor.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")
foreach (TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_TARGET ${TEST_FILE} NAME_WE)
    add_executable(${TEST_TARGET}
                   ${TEST_FILE}
                   ${COMMON_SRC_FILES}
                   ${PREPROCESS_SRC_FILES})
    target_link_libraries(${TEST_TARGET}
                          pp_proto_defs
                          ${GTEST_BOTH_LIBRARIES}
                          gflags
                          ${Protobuf_LIBRARIES}
                          ${PCL_LIBRARIES}
                          ${Boost_LIBRARIES})
    add_test(NAME ${TEST_FILE} COMMAND ${TEST_TARGET}
             "--test_base=${CMAKE_CURRENT_SOURCE_DIR}/test_base")
    add_dependencies(${TEST_TARGET} pp_proto_defs)
endforeach()

set(TEST_EXAMPLE_GEN test_example_gen)
add_executable(${TEST_EXAMPLE_GEN}
               ${CMAKE_CURRENT_SOURCE_DIR}/test_example_gen.cpp
               ${COMMON_SRC_FILES}
               ${PREPROCESS_SRC_FILES})
target_link_libraries(${TEST_EXAMPLE_GEN}
                      pp_proto_defs
                      gflags
                      ${Protobuf_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${Boost_LIBRARIES})
add_dependencies(${TEST_EXAMPLE_GEN} pp_proto_defs)