find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS "Boost_FILESYSTEM_LIBRARIES: ${Boost_FILESYSTEM_LIBRARIES}")