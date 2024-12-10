#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void test01() {
    enum {
        WARP_PER_THREAD = 32,
        THREADS_PER_BLOCK = 128,
        THREADS_PER_SM = 1024
    };
    int warp_ID_Value = 0;
    warp_ID_Value = threadIdx.x / WARP_PER_THREAD;
    printf("\nThe block ID is %d --- The thread ID is %d --- The warp ID is %d", blockIdx.x, threadIdx.x, warp_ID_Value);
}

__global__ void test02(int *a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    a[tid] = tid;
    printf("\na[%d] = %d", tid, a[tid]);
}
__global__ void test03() {
    printf("\nHello World from GPU!\n");
}

__global__ void test04() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("\nThe block ID is %d --- The thread ID is %d --- tid is %d", blockIdx.x, threadIdx.x, tid);
}

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    enum {
        E_OK = 0,
        E_FAIL = 1
    };
    /* kernel name <<< num_of_blocks, num_of_threads_per_block >>> (); */
    test01<<<2, 64>>>();

    /* wait for kernel completion */
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaSuccess != cudaStatus) {
        printf("test01 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }
    /* host message */
    printf("\nGPU completed pararell processes! Congratulations from main()\n");

    /* kernel name <<< num_of_blocks, num_of_threads_per_block >>> (parameters); */
    int h_a[128];
    test02<<<2, 64>>>(h_a);
    /* wait for kernel completion */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaSuccess != cudaStatus) {
        printf("test02 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }
    /* host message */
    printf("\nGPU completed pararell processes! Congratulations from main()\n");
    test03<<<1, 1>>>();
    cudaStatus = cudaDeviceSynchronize();
    if (cudaSuccess != cudaStatus) {
        printf("\ntest03 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }

    test04<<<2, 64>>>();
    cudaStatus = cudaDeviceSynchronize();
    if (cudaSuccess != cudaStatus) {
        printf("\ntest04 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }

    return E_OK;
}