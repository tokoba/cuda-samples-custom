# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.18)

# プロジェクト名と言語を指定
project(CUDA_VECTOR LANGUAGES CXX CUDA)

# CUDA Toolkitの検出
find_package(CUDAToolkit REQUIRED)
# ソースファイルをリストアップ
set(SOURCES
    ./src/vectorAdd.cu
)

# 実行ファイルの生成
add_executable(vector_add ${SOURCES})

# CUDA Toolkitのリンク
# target_link_libraries(cuda_example PRIVATE CUDAToolkit::cuda)

# コンパイルオプションの設定（必要に応じて）
# RTX3090 : 8.6 Compute Capability, RTX4090 8.9 Compute Capability
set(CUDA_ARCH "sm_86")  # 使用するGPUアーキテクチャを指定（例: sm_70 for Turing）
target_compile_options(vector_add PRIVATE -arch=${CUDA_ARCH})

# オプション：デバッグ情報の追加
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vector_add PRIVATE -g)
endif()

# オプション：最適化レベルの設定（リリースビルドの場合）
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     target_compile_options(vector_add PRIVATE -O3)
# endif()

