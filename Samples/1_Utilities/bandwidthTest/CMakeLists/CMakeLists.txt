# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.18)

# プロジェクト名と言語を指定
project(CUDABandWidthTest LANGUAGES CXX CUDA)

# CUDA Toolkitの検出
find_package(CUDAToolkit REQUIRED)

# ソースファイルをリストアップ
set(SOURCES
    ./src/bandwidthTest.cu
)

set(HEADERS_DIR ${CMAKE_SOURCE_DIR}/../../Common/)  # ヘッダーファイルが存在するディレクトリ

# ソースファイルからオブジェクトファイルを生成するカスタムコマンド
foreach(SOURCE_FILE IN LISTS SOURCES)
    get_filename_component(BASE_NAME ${SOURCE_FILE} NAME_WE)  # ファイル名の拡張子なし部分を取得
    set(OBJECT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}.o")  # オブジェクトファイルのパス

    add_custom_command(
        OUTPUT ${OBJECT_FILE}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
        -arch=${CUDA_ARCH} -c ${SOURCE_FILE}
        -o ${OBJECT_FILE}
        DEPENDS ${SOURCE_FILE}
        COMMENT "Compiling ${SOURCE_FILE} to ${OBJECT_FILE}"
    )

    # オブジェクトファイルをリストに追加
    list(APPEND OBJECT_FILES ${OBJECT_FILE})
endforeach()

# 実行ファイルの生成
add_executable(bandwidthTest ${SOURCES})

# インクルードディレクトリの設定
target_include_directories(bandwidthTest PRIVATE ${HEADERS_DIR})

# コンパイルオプションの設定（必要に応じて）
# RTX3090 : 8.6 Compute Capability, RTX4090 8.9 Compute Capability
set(CUDA_ARCH "sm_86")  # 使用するGPUアーキテクチャを指定（例: sm_70 for Turing）
target_compile_options(bandwidthTest PRIVATE -arch=${CUDA_ARCH})

# オプション：デバッグ情報の追加
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(bandwidthTest PRIVATE -g)
endif()

# オプション：最適化レベルの設定（リリースビルドの場合）
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     target_compile_options(cuda_example PRIVATE -O3)
# endif()

