
#include <hip/hip_runtime.h>
extern "C" __global__ void add_scalar(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { data[i] += scalar; }
}
