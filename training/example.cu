// example.cu
#include <iostream>

__global__ void add(int *a, int *b, int *c, int N) {
    // �O���[�o���X���b�hID���v�Z
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {  // �͈̓`�F�b�N
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 1024;  // �v�f��
    
    size_t bytes = N * sizeof(int);
    cudaSetDevice(0);

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // �z�X�g�������̊m��
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // �f�[�^�̏�����
    for (int i = 0; i < N; ++i) {
        h_a[i] = -i;
        h_b[i] = i * i;
    }

    // �f�o�C�X�������̊m��
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // �z�X�g����f�o�C�X�ւ̃f�[�^�]��
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // �J�[�l���̎��s
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // �G���[�`�F�b�N
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // �f�o�C�X����z�X�g�ւ̃f�[�^�]��
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // ���ʂ̊m�F
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": expected " << h_a[i] + h_b[i] << ", got " << h_c[i] << std::endl;
            return 1;
        }
    }

    // �������̉��
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "All tests passed!" << std::endl;

    return 0;
}