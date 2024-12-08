#include <cuda_runtime.h>
#include <iostream>

// CUDA �J�[�l���֐�: �z��̗v�f���Ƃ̉��Z
__global__ void add(int* a, int* b, int* c, size_t n) {
    // �u���b�N���̃X���b�h ID ���擾
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Warp ���ł̃X���b�h ID ���v�Z (�����̗p�r�̂��߂ɃR�����g�A�E�g)
    // unsigned int warpId = tid / 32;
    // unsigned int laneId = tid % 32;

    // �z��͈͓̔��ł���Ή��Z���s��
    if (tid < n) {
        c[tid] = a[tid] + b[tid];

        // Warp ���ł̓����������R�����g
        // ���� Warp ���̃X���b�h�͓������߂𓯎��Ɏ��s���邽�߁A
        // ���̓_�œ������ۂ����B
    }
}

int main() {
    const size_t n = 1024;
    const size_t bytes = n * sizeof(int);

    // �z�X�g�������̊m�ۂƏ�����
    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_c = new int[n];

    for (size_t i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // �f�o�C�X�������̊m��
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // �z�X�g����f�o�C�X�ւ̃f�[�^�]��
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // �J�[�l���̎��s: 128 �X���b�h�̃u���b�N���g�p
    size_t threadsPerBlock = 128;
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // �f�o�C�X����z�X�g�ւ̃f�[�^�]��
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // ���ʂ̊m�F
    for (size_t i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": expected " << h_a[i] + h_b[i] << ", got " << h_c[i] << std::endl;
            return 1;
        }
    }

    // �������̉��
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "All tests passed!" << std::endl;

    return 0;
}