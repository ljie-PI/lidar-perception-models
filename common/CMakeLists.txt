cmake_minimum_required(VERSION 3.10)
project(perception_models_common)

include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Eigen.cmake)

include_directories(${CMAKE_SOURCE_DIR}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")
foreach (TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_TARGET ${TEST_FILE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_FILE})
    target_link_libraries(${TEST_TARGET}
                          ${GTEST_BOTH_LIBRARIES}
                          ${Boost_LIBRARIES})
    add_test(NAME ${TEST_FILE} COMMAND ${TEST_TARGET})
    message(STATUS "ADDED TEST: ${TEST_TARGET}")
endforeach()