cmake_minimum_required(VERSION 3.15)
project(NeuralGraphCompiler)

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
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(ng_compiler main.cpp)
add_subdirectory(front)
add_subdirectory(codegen)

target_link_libraries(ng_compiler PRIVATE
    frontend_lib
    codegen_lib
)

#cmake -B build -S . -G Ninja \
  #-DMLIR_DIR=/mnt/c/prplus/NeuralGraph/codegen/llvm-project/build/lib/cmake/mlir \
  #-DLLVM_DIR=/mnt/c/prplus/NeuralGraph/codegen/llvm-project/build/lib/cmake/llvm
