#include <cuda_runtime.h>
#include <iostream>

// CUDA カーネル関数: 配列の要素ごとの加算
__global__ void add(int* a, int* b, int* c, size_t n) {
    // ブロック内のスレッド ID を取得
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Warp 内でのスレッド ID を計算 (将来の用途のためにコメントアウト)
    // unsigned int warpId = tid / 32;
    // unsigned int laneId = tid % 32;

    // 配列の範囲内であれば加算を行う
    if (tid < n) {
        c[tid] = a[tid] + b[tid];

        // Warp 内での同期を示すコメント
        // 同じ Warp 内のスレッドは同じ命令を同時に実行するため、
        // この点で同期が保たれる。
    }
}

int main() {
    const size_t n = 1024;
    const size_t bytes = n * sizeof(int);

    // ホストメモリの確保と初期化
    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_c = new int[n];

    for (size_t i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // デバイスメモリの確保
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ホストからデバイスへのデータ転送
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // カーネルの実行: 128 スレッドのブロックを使用
    size_t threadsPerBlock = 128;
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // デバイスからホストへのデータ転送
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 結果の確認
    for (size_t i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": expected " << h_a[i] + h_b[i] << ", got " << h_c[i] << std::endl;
            return 1;
        }
    }

    // メモリの解放
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "All tests passed!" << std::endl;

    return 0;
}