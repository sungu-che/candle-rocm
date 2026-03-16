#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL(vector_add, dim3(grid_size), dim3(block_size), 0, 0, d_a, d_b, d_c, N);

    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            printf("FAIL at %d: got %f expected %f\n", i, h_c[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    if (errors == 0) {
        printf("PASS: vector_add %d elements correct\n", N);
    }

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors > 0 ? 1 : 0;
}
