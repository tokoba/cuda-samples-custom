# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.18)

# プロジェクト名と言語を指定
project(CUDAWarp LANGUAGES CXX CUDA)

# CUDA Toolkitの検出
find_package(CUDAToolkit REQUIRED)
# if(CUDA_FOUND)
# 	message("Cuda version: " ${CUDA_VERSION})
# 	CUDA_ADD_EXECUTABLE(Main
# 		main.cu
# 	)
# endif(
#     message("[CMake ERROR] Cuda not found")
# )

# ソースファイルをリストアップ
set(SOURCES
    ./src/cuda_warp2.cu
)

# 実行ファイルの生成
add_executable(cuda_warp ${SOURCES})

# CUDA Toolkitのリンク
# target_link_libraries(cuda_example PRIVATE CUDAToolkit::cuda)

# コンパイルオプションの設定（必要に応じて）
# RTX3090 : 8.6 Compute Capability, RTX4090 8.9 Compute Capability
set(CUDA_ARCH "sm_86")  # 使用するGPUアーキテクチャを指定（例: sm_70 for Turing）
target_compile_options(cuda_warp PRIVATE -arch=${CUDA_ARCH})

# オプション：デバッグ情報の追加
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_warp PRIVATE -g)
endif()

# オプション：最適化レベルの設定（リリースビルドの場合）
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     target_compile_options(cuda_warp PRIVATE -O3)
# endif()

