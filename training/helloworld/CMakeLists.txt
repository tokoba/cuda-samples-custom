# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.18)

# プロジェクト名と言語を指定
project(CUDAExample LANGUAGES CXX CUDA)

# CUDA Toolkitの検出
find_package(CUDAToolkit REQUIRED)

# ソースファイルをリストアップ
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/helloworld.cu
)

message("cmake current source directory is ", ${CMAKE_CURRENT_SOURCE_DIR})

# 実行ファイルの生成
add_executable(cuda_example ${SOURCES})

# CUDA Toolkitのリンク
# target_link_libraries(cuda_example PRIVATE CUDAToolkit::cuda)

# コンパイルオプションの設定（必要に応じて）
# RTX3090 : 8.6 Compute Capability, RTX4090 8.9 Compute Capability
set(CUDA_ARCH "sm_86")  # 使用するGPUアーキテクチャを指定（例: sm_70 for Turing）
target_compile_options(cuda_example PRIVATE -arch=${CUDA_ARCH})

# オプション：デバッグ情報の追加
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_example PRIVATE -g)
endif()

# オプション：最適化レベルの設定（リリースビルドの場合）
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     target_compile_options(cuda_example PRIVATE -O3)
# endif()

# ビルドディレクトリを ${CMAKE_CURRENT_SOURCE_DIR}/build に設定
# 実行ファイルの出力先を設定
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
message("Build files will be written to: " ${BUILD_DIR})

set_target_properties(cuda_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}
)

# CMakeのビルドディレクトリを設定（オプション）
if(NOT EXISTS "${BUILD_DIR}")
    file(MAKE_DIRECTORY "${BUILD_DIR}")
endif()