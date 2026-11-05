// Matched f32 GEMM study for Goose benchmarks/goose/gemm_study.goose.
// clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror
// CLI: ./gemm_study <ijk|ikj|blocked32|rows> <n> <repetitions>
// Output: GEMM_RESULT,c,variant,n,repetitions,elapsed_ns,max_abs_error,checksum
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t time_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) { perror("clock_gettime"); exit(1); }
    return (int64_t)t.tv_sec * 1000000000LL + (int64_t)t.tv_nsec;
}

static void gemm_ijk(const float *a, const float *b, float *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < n; ++k)
                acc += a[i * n + k] * b[k * n + j];
            c[i * n + j] = acc;
        }
}

static void gemm_ikj(const float *a, const float *b, float *c, size_t n) {
    for (size_t p = 0; p < n * n; ++p) c[p] = 0.0f;
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            const float aik = a[i * n + k];
            for (size_t j = 0; j < n; ++j)
                c[i * n + j] += aik * b[k * n + j];
        }
}

static void gemm_blocked32(const float *a, const float *b, float *c, size_t n) {
    for (size_t p = 0; p < n * n; ++p) c[p] = 0.0f;
    const size_t tiles = (n + 31) / 32;
    for (size_t ti = 0; ti < tiles; ++ti) {
        const size_t i0 = ti * 32, i1 = i0 + 32 < n ? i0 + 32 : n;
        for (size_t tk = 0; tk < tiles; ++tk) {
            const size_t k0 = tk * 32, k1 = k0 + 32 < n ? k0 + 32 : n;
            for (size_t tj = 0; tj < tiles; ++tj) {
                const size_t j0 = tj * 32, j1 = j0 + 32 < n ? j0 + 32 : n;
                for (size_t i = i0; i < i1; ++i)
                    for (size_t k = k0; k < k1; ++k) {
                        const float aik = a[i * n + k];
                        for (size_t j = j0; j < j1; ++j)
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
        }
    }
}

static void gemm_rows(const float *a, const float *b, float *c, size_t n) {
    for (size_t pos=0; pos<n*n; ++pos) c[pos] = 0.0f;
    for (size_t i=0; i<n; ++i) {
        const float *a_row = a + i*n;
        float *c_row = c + i*n;
        for (size_t k=0; k<n; ++k) {
            const float *b_row = b + k*n;
            const float aik = a_row[k];
            for (size_t j=0; j<n; ++j) c_row[j] += aik * b_row[j];
        }
    }
}

static void perform(const float *a, const float *b, float *c, size_t n,
                    const char *variant) {
    if (strcmp(variant, "ijk") == 0) gemm_ijk(a, b, c, n);
    else if (strcmp(variant, "ikj") == 0) gemm_ikj(a, b, c, n);
    else if (strcmp(variant, "blocked32") == 0) gemm_blocked32(a, b, c, n);
    else gemm_rows(a, b, c, n);
}

static long parse_arg(const char *raw, long limit) {
    char *end = NULL;
    errno = 0;
    const long n = strtol(raw, &end, 10);
    if (errno || end == raw || *end != 0 || n < 1 || n > limit) {
        fprintf(stderr, "invalid integer: %s\n", raw);
        exit(2);
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc != 4 ||
        (strcmp(argv[1], "ijk") && strcmp(argv[1], "ikj") &&
         strcmp(argv[1], "blocked32") && strcmp(argv[1], "rows"))) {
        fprintf(stderr, "usage: gemm_study <ijk|ikj|blocked32|rows> <n> <repetitions>\n");
        return 2;
    }
    const char *variant = argv[1];
    const size_t n = (size_t)parse_arg(argv[2], 256);
    const long repetitions = parse_arg(argv[3], 10000);
    const size_t count = n * n;
    float *a = malloc(count * sizeof(float));
    float *b = malloc(count * sizeof(float));
    float *c = malloc(count * sizeof(float));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a); free(b); free(c);
        return 2;
    }
    for (size_t i = 0; i < count; ++i) {
        a[i] = ((float)(i % 257) - 128.0f) * 0.00713f;
        b[i] = ((float)(i % 199) - 99.0f) * 0.01517f;
        c[i] = 0.0f;
    }
    perform(a, b, c, n, variant);
    const int64_t start = time_ns();
    for (long r = 0; r < repetitions; ++r)
        perform(a, b, c, n, variant);
    const int64_t elapsed = time_ns() - start;

    double max_abs_error = 0.0, checksum = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += (double)a[i * n + k] * (double)b[k * n + j];
            const double actual = (double)c[i * n + j];
            const double error = fabs(actual - expected);
            const double tol = 0.0005 + 0.0001 * fabs(expected);
            if (error > tol) {
                fprintf(stderr,
                        "element mismatch: n=%zu i=%zu j=%zu got=%.9g ref=%.9g error=%.9g\n",
                        n, i, j, actual, expected, error);
                free(a); free(b); free(c);
                return 3;
            }
            if (error > max_abs_error) max_abs_error = error;
            checksum += actual;
        }
    printf("GEMM_RESULT,c,%s,%zu,%ld,%" PRId64 ",%.17g,%.17g\n",
           variant, n, repetitions, elapsed, max_abs_error, checksum);
    free(a); free(b); free(c);
    return 0;
}
