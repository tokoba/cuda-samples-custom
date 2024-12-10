#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// __global__ void addKernel(int *c, const int *a, const int *b) {
//     int i = threadIdx.x;
//     c[i] = a[i] + b[i];
// }

__global__ void test01() {
    int warp_ID_Value = 0;
    warp_ID_Value = threadIdx.x / 32;
    printf("\nThe block ID is %d --- The thread ID is %d --- The warp ID is %d", blockIdx.x, threadIdx.x, warp_ID_Value);
}

int main() {
    enum {
        E_OK = 0,
        E_FAIL = 1
    };
    /* kernel name <<< num_of_blocks, num_of_threads_per_block >>> (); */
    test01<<<32, 64>>>();

    /* wait for kernel completion */
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaSuccess != cudaStatus) {
        printf("test01 cudaStatus Failure, cudaDeviceSynchronize failed.\n");
        return E_FAIL;
    }
    /* host message */
    printf("\nGPU completed pararell processes! Congratulations from main()\n");

    return E_OK;
    // const int arraySize = 5;
    // const int a[arraySize] = {1, 2, 3, 4, 5};
    // const int b[arraySize] = {10,
    //                           20,
    //                           30,
    //                           40,
    //                           50};
    // int c[arraySize] = {0};
    // enum {
    //     E_FAIL = 1
    // };

    // /* Add vectors in parallel */
    // cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    // if (cudaSuccess != cudaStatus) {
    //     fprintf(stderr, "addWithCuda failed\n");
    //     return E_FAIL;
    // }

    // printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //        c[0], c[1], c[2], c[3], c[4]);

    // /* cudaDeviceReset must be called before exiting
    //  * in order for profiling and tracing tools such as NSight
    //  * and Visual Profiler to show complete traces.
    //  */
    // cudaStatus = cudaDeviceReset();
    // if (cudaSuccess != cudaStatus) {
    //     fprintf(stderr, "cudaDeviceReset failed!\n");
    // }
}