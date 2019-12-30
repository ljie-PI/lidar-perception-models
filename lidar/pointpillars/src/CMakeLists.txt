cmake_minimum_required(VERSION 2.8)

include(${CMAKE_SOURCE_DIR}/cmake/GFlags.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PCL.cmake)

include_directories(${CMAKE_SOURCE_DIR}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

## generate pointpillars examples
set(PP_EXAMPLE_GEN pp_example_gen)
add_executable(${PP_EXAMPLE_GEN}
        ${CMAKE_CURRENT_SOURCE_DIR}/pp_example_gen.cpp)
