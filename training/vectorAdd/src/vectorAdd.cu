#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"

#define SIZE (2048) /* Define the size of the vectors */

#define TPW (32) /* Threads per Warp (FIXED) */

typedef enum tag_GPU_PRODUCTS {
    RTX4090,
    RTX4080,
    RTX3090,
    MAX_GPU_PRODUCTS
} GPU_PRODUCTS;

/* SM is Streaming Multiprocessor */
/* GPU Hierarcy
 * Grid == GPU itself (all the computational resources)
 * Block == SM - Streaming Multiprocessor (each calculation block)
 * Thread == Warp - 32 threads
 * Core == INT32, FP32 etc(each ALU)
 *
 */
typedef struct tag_GPU_SPEC {
    int cudaCoresPerGrid;   /* CUDA Cores per Grid */
    int tensorCoresPerGrid; /* Tensor Cores per Grid */
    int blockPerGrid;       /* Block per Grid (SM per Grid) */
    int cudaCoresPerSM;     /* CUDA Cores per SM */
    int tensorCoresPerSM;   /* Tensor Cores per SM */
} GPU_SPEC;                 /* Define the GPU Spec */

GPU_SPEC gpu_spec[MAX_GPU_PRODUCTS] = {
    {
        /* RTX4090 (AD102) */
        16384, /* CUDA Core */
        512,   /* Tensor Core */
        128,   /* Block per Grid (SM per Grid) */
        128,   /* CUDA Cores per SM */
        4      /* Tensor Cores per SM */

    },
    {
        /* RTX4080 (AD103) */
        9728, /* CUDA Core */
        304,  /* Tensor Core */
        76,   /* Block per Grid (SM per Grid) */
        128,  /* CUDA Cores per SM */
        4     /* Tensor Cores per SM */
    },
    {
        /* RTX3090 (GB102) */
        10496, /* CUDA Core */
        328,   /* Tensor Core */
        82,    /* Block per Grid (SM per Grid) */
        128,   /* CUDA Cores per SM */
        4      /* Tensor Cores per SM */
    }

}; /* Define the GPU Spec */

/* CUDA Kernel function for vector addition */
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure the index is within bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    /* Allocate memory space */
    int *a, *b, *c;        // Host pointers
    int *d_a, *d_b, *d_c;  // Device pointers
    int size = SIZE * sizeof(int);
    cudaEvent_t start, stop;
    cudaEventCreate(&start, 0);
    cudaEventCreate(&stop, 0);

    cudaError_t cudaStatus = cudaSuccess;
    enum {
        E_OK = 0,
        E_FAIL = 1
    };

    // Allocate memory on host and initialize
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    /* Allocate device vectors */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    /* initialize the inputs */
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = SIZE - i;
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    /* Launch the Vector Add Cuda Kernel 2 blocks*/
    cudaEventRecord(start);
    // vectorAdd<<<2, 1024>>>(d_a, d_b, d_c, SIZE);
    vectorAdd<<<128, 16>>>(d_a, d_b, d_c, SIZE);
    cudaEventRecord(stop);

    /* Copy result back to host */
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    /* wait for kernel completion */
    cudaStatus = cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f msec\n", milliseconds);
    if (cudaSuccess != cudaStatus) {
        printf("test02 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }
    printf("Execution finished.\n");
/* Verify the result */
#if 0
    for (int i = 0; i < SIZE; i++) {
        printf("index %d: %d + %d expected %d, got %d\n", i, a[i], b[i], a[i] + b[i], c[i]);
    }
#endif
    /* Free device global memory */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    /* Free host memory */
    free(a);
    free(b);
    free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Memory freed.\n");

    return 0;
}