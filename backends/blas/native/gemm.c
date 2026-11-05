#include <cblas.h>
#include <limits.h>
#include <stdint.h>

// A stable plain-C ABI; arrays are borrowed only for the duration of the call.
int32_t deep_goose_blas_sgemm(const float *a, const float *b, float *c,
                             int64_t m, int64_t k, int64_t n) {
    if (!a || !b || !c || m <= 0 || k <= 0 || n <= 0 ||
        m > INT_MAX || k > INT_MAX || n > INT_MAX)
        return -1;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f, a, (int)k, b, (int)n, 0.0f, c, (int)n);
    return 0;
}
