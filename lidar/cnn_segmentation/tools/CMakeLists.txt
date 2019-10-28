cmake_minimum_required(VERSION 2.8)
project(cnnseg_tools)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -pthread")

include(${CMAKE_SOURCE_DIR}/cmake/Gears.cmake)

set(3RD_ROOT ${CMAKE_SOURCE_DIR}/../../../third_party)

add_subdirectory(src)
