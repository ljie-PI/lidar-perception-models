cmake_minimum_required(VERSION 2.8)
project(cnnseg_tools)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O2 -fPIE -fPIC -msse4.1")

add_subdirectory(src)
