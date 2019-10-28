set(GFLAGS_FOUND 1 CACHE INTERNAL "GFLAGS_FOUND")
set(GFLAGS_ROOT ${3RD_ROOT}/gflags CACHE INTERNAL "(GFLAGS_ROOT")
set(GFLAGS_INCLUDE_DIRS ${GFLAGS_ROOT}/include CACHE INTERNAL "GFLAGS_INCLUDE_DIRS")
set(GFLAGS_LIBRARY_DIRS ${GFLAGS_ROOT}/lib CACHE INTERNAL "GFLAGS_LIBRARY_DIRS")
file(GLOB LIBRARIES "${GFLAGS_LIBRARY_DIRS}/*.so")
set(GFLAGS_LIBRARIES ${LIBRARIES} CACHE INTERNAL "GFLAGS_LIBRARIES")

include_directories(${GFLAGS_INCLUDE_DIRS})
link_directories(${GFLAGS_LIBRARY_DIRS})
