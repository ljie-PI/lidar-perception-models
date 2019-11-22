cmake_minimum_required(VERSION 2.8)

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
