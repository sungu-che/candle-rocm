#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <math.h>

int main() {
    // 2x2 matmul: C = A * B
    // A = [[1, 2], [3, 4]]  B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    float h_A[] = {1, 3, 2, 4};  // column-major
    float h_B[] = {5, 7, 6, 8};  // column-major
    float h_C[] = {0, 0, 0, 0};

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, 4 * sizeof(float));
    hipMalloc(&d_B, 4 * sizeof(float));
    hipMalloc(&d_C, 4 * sizeof(float));

    hipMemcpy(d_A, h_A, 4 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, 4 * sizeof(float), hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    float alpha = 1.0f, beta = 0.0f;
    rocblas_status status = rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        2, 2, 2, &alpha, d_A, 2, d_B, 2, &beta, d_C, 2);

    if (status != rocblas_status_success) {
        printf("FAIL: rocblas_sgemm returned status %d\n", status);
        rocblas_destroy_handle(handle);
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        return 1;
    }

    hipMemcpy(h_C, d_C, 4 * sizeof(float), hipMemcpyDeviceToHost);

    // h_C is column-major: [19, 43, 22, 50]
    float expected[] = {19, 43, 22, 50};
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(h_C[i] - expected[i]) > 1e-4) {
            printf("FAIL at %d: got %f expected %f\n", i, h_C[i], expected[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS: rocblas_sgemm 2x2 correct\n");
    }

    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return errors > 0 ? 1 : 0;
}
