# Locate and configure the Gears(third party packages).
#
# Defines the following variables:
#
#   GEARS_ROOT - Then root directory for gears
#   GEARS_ARCH - Then directory for a special architecture
#   GEARS_LIBRARY_DIR - The library directory for all gears
#   GEARS_INCLUDE_DIR - The include directory for all gears
#   GEARS_BIN_DIR - The binary directory for all gears
#
#  ====================================================================
set(GEARS_ROOT ${CMAKE_SOURCE_DIR}/../gears CACHE INTERNAL "GEARS_ROOT")
set(GEARS_ARCH ${GEARS_ROOT}/${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL "GEARS_ARCH")
set(GEARS_LIBRARY_DIR ${GEARS_ARCH}/lib CACHE INTERNAL "GEARS_LIBRARY_DIR")
set(GEARS_INCLUDE_DIR ${GEARS_ARCH}/include CACHE INTERNAL "GEARS_INCLUDE_DIR")
set(GEARS_BIN_DIR ${GEARS_ARCH}/bin CACHE INTERNAL "GEARS_BIN_DIR")
include_directories(${GEARS_INCLUDE_DIR})
link_directories(${GEARS_LIBRARY_DIR})
