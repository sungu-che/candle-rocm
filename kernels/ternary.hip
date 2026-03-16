#include "hip_utils.h"

// where_cond: out[i] = cond[i] ? t[i] : f[i]
// All three inputs share the same info (dims/strides) when non-contiguous.
extern "C" __global__ void where_cond_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const uint8_t *cond,
    const float *t,
    const float *f,
    float *out
) {
    const size_t *dims = info;
    const size_t *strides = info ? info + num_dims : nullptr;
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides);
        out[i] = cond[src_i] ? t[src_i] : f[src_i];
    }
}
