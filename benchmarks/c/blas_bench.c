// OpenBLAS C baseline for the Goose native adapter, matching all input values.
#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#include <stdlib.h>
#ifdef DEEP_GOOSE_ADAPTER_BENCH
extern int32_t deep_goose_blas_sgemm(const float *a, const float *b, float *c,
                                     int64_t m, int64_t k, int64_t n);
#define IMPL_NAME "c_blas_adapter"
static void call_gemm(const float *a, const float *b, float *c, int n) {
    if (deep_goose_blas_sgemm(a, b, c, n, n, n) != 0) abort();
}
#else
#define IMPL_NAME "c_blas"
static void call_gemm(const float *a, const float *b, float *c, int n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                1.0f, a, n, b, n, 0.0f, c, n);
}
#endif

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
int main(int argc, char **argv) {
    if (argc != 4 || strcmp(argv[1],"matmul")) return 2;
    int n = atoi(argv[2]), iterations = atoi(argv[3]);
    if (n <= 0 || n > 1024 || iterations <= 0) return 2;
    size_t count = (size_t)n*n;
    float *a=malloc(count*sizeof(float));
    float *b=malloc(count*sizeof(float));
    float *c=malloc(count*sizeof(float));
    if (!a || !b || !c) return 3;
    for (size_t i=0; i<count; ++i) {
        a[i]=((float)(i%257)-128.0f)*0.0078125f;
        b[i]=((float)(i%199)-99.0f)*0.015625f;
        c[i]=0.0f;
    }
    call_gemm(a, b, c, n);
    int64_t before=ns();
    for (int it=0; it<iterations; ++it)
        call_gemm(a, b, c, n);
    int64_t elapsed=ns()-before;
    double checksum=0;
    for (size_t i=0; i<count; ++i) checksum+=(double)c[i];
    printf("RESULT,%s,matmul,%d,%d,%" PRId64 ",%.17g\n",
           IMPL_NAME,n,iterations,elapsed,checksum);
    free(a);free(b);free(c);
    return 0;
}
