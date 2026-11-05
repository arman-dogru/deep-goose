// Baseline C numerical kernels, matched to benchmarks/goose/cpu_bench.goose.
// Compile: cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -o cpu_bench cpu_bench.c
// Usage: ./cpu_bench <add|relu|dot|matmul|matmul_ikj> <n> <iterations>
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t monotonic_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) {
        perror("clock_gettime");
        exit(1);
    }
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
static void vector_add(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void vector_relu(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = a[i] + b[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}
static float vector_dot(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
static void matrix_multiply(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++) sum += a[i*n + k] * b[k*n + j];
            out[i*n+j] = sum;
        }
    }
}
// Loop-reordered GEMM with contiguous writes in the innermost loop.
static void matrix_multiply_ikj(const float *a, const float *b, float *out, size_t n) {
    for (size_t pos=0; pos<n*n; ++pos) out[pos] = 0.0f;
    for (size_t i=0; i<n; ++i)
        for (size_t k=0; k<n; ++k) {
            float v = a[i*n+k];
            for (size_t j=0; j<n; ++j) out[i*n+j] += v * b[k*n+j];
        }
}
static size_t parse_positive(const char *raw) {
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(raw, &end, 10);
    if (errno || end == raw || *end || n == 0 || n > 1048576) {
        fprintf(stderr, "invalid positive integer: %s\n", raw);
        exit(2);
    }
    return (size_t)n;
}
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: cpu_bench <add|relu|dot|matmul|matmul_ikj> <n> <iterations>\n");
        return 2;
    }
    const char *op = argv[1];
    int is_add = !strcmp(op, "add");
    int is_relu = !strcmp(op, "relu");
    int is_dot = !strcmp(op, "dot");
    int is_matmul = !strcmp(op, "matmul");
    int is_ikj = !strcmp(op, "matmul_ikj");
    if (!(is_add || is_relu || is_dot || is_matmul || is_ikj)) {
        fprintf(stderr, "unknown operation\n");
        return 2;
    }
    size_t n = parse_positive(argv[2]);
    size_t iterations = parse_positive(argv[3]);
    if ((is_matmul || is_ikj) && n > 4096) {
        fprintf(stderr, "matmul dimension exceeds safety limit\n");
        return 2;
    }
    size_t count = (is_matmul || is_ikj) ? n*n : n;
    if (count > SIZE_MAX / sizeof(float)) return 2;
    float *a = malloc(count * sizeof(float));
    float *b = malloc(count * sizeof(float));
    float *out = malloc(count * sizeof(float));
    if (!a || !b || !out) {
        fprintf(stderr, "allocation failure\n");
        free(a); free(b); free(out);
        return 2;
    }
    for (size_t i=0; i<count; i++) {
        a[i] = ((float)(i % 257) - 128.0f) * 0.0078125f;
        b[i] = ((float)(i % 199) - 99.0f) * 0.015625f;
        out[i] = 0.0f;
    }
    float last_dot = 0.0f;
    if (is_add) vector_add(a,b,out,count);
    if (is_relu) vector_relu(a,b,out,count);
    if (is_dot) last_dot = vector_dot(a,b,count);
    if (is_matmul) matrix_multiply(a,b,out,n);
    if (is_ikj) matrix_multiply_ikj(a,b,out,n);

    int64_t start = monotonic_ns();
    for (size_t r=0; r<iterations; r++) {
        if (is_add) vector_add(a,b,out,count);
        if (is_relu) vector_relu(a,b,out,count);
        if (is_dot) last_dot = vector_dot(a,b,count);
        if (is_matmul) matrix_multiply(a,b,out,n);
    if (is_ikj) matrix_multiply_ikj(a,b,out,n);
    }
    int64_t elapsed = monotonic_ns() - start;

    double checksum = (double)last_dot;
    if (!is_dot) {
        checksum = 0.0;
        for (size_t i=0; i<count; i++) checksum += (double)out[i];
    }
    printf("RESULT,c,%s,%zu,%zu,%" PRId64 ",%.17g\n",
           op, n, iterations, elapsed, checksum);
    free(a); free(b); free(out);
    return 0;
}
