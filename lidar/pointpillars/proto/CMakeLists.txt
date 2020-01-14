cmake_minimum_required(VERSION 3.10)

include(${CMAKE_SOURCE_DIR}/cmake/Protobuf.cmake)

file(GLOB PROTO_DEFS ${CMAKE_CURRENT_SOURCE_DIR}/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_DEFS})
add_library(pp_proto_defs SHARED ${PROTO_SRCS})

protobuf_generate_python(PROTO_PY_SRCS ${PROTO_DEFS})
add_custom_target(pp_proto_python ALL DEPENDS ${PROTO_PY_SRCS})
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generated/__init__.py)
install(FILES ${PROTO_PY_SRCS} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generated)
