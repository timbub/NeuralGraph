cmake_minimum_required(VERSION 3.14)
project(NeuralGraph)

add_compile_options(
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

add_link_options(
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

find_package(Protobuf REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/inc)
include_directories(${CMAKE_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_SOURCE_DIR}/onnx/onnx.proto)

add_executable(graph
    src/main.cpp
    src/parser.cpp
    src/visualization.cpp
    ${PROTO_SRCS}
)
target_link_libraries(graph ${Protobuf_LIBRARIES})
