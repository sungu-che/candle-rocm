#include "hip_utils.h"

// affine: out[i] = inp[i] * mul + add
// info is null for contiguous, otherwise points to [dims..., strides...]
extern "C" __global__ void affine_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    const float mul,
    const float add
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, info, info + num_dims);
        out[i] = inp[src_i] * mul + add;
    }
}
