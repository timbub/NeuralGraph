protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto)

add_library(frontend_lib STATIC
    src/parser.cpp
    src/visualization.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(frontend_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(frontend_lib PUBLIC ${Protobuf_LIBRARIES})
