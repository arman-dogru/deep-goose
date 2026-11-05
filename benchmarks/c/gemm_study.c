// Matched f32 GEMM study for Goose benchmarks/goose/gemm_study.goose.
// clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror
// CLI: ./gemm_study <ijk|ikj|blocked32|rows|micro4|micro1x4> <n> <repetitions>
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

// Four-by-four register blocking. Every output follows k-ascending
// accumulation; rows/columns not divisible by four use row traversal.
static void gemm_micro4(const float *a, const float *b, float *c, size_t n) {
    for (size_t pos=0; pos<n*n; ++pos) c[pos] = 0.0f;
    const size_t tiles = n/4, full = tiles*4;
    for (size_t bi=0; bi<tiles; ++bi) {
        const size_t i=bi*4;
        const float *ar0 = a + (i+0)*n;
        const float *ar1 = a + (i+1)*n;
        const float *ar2 = a + (i+2)*n;
        const float *ar3 = a + (i+3)*n;
        for (size_t bj=0; bj<tiles; ++bj) {
            const size_t j=bj*4;
            float s00 = 0.0f;
            float s01 = 0.0f;
            float s02 = 0.0f;
            float s03 = 0.0f;
            float s10 = 0.0f;
            float s11 = 0.0f;
            float s12 = 0.0f;
            float s13 = 0.0f;
            float s20 = 0.0f;
            float s21 = 0.0f;
            float s22 = 0.0f;
            float s23 = 0.0f;
            float s30 = 0.0f;
            float s31 = 0.0f;
            float s32 = 0.0f;
            float s33 = 0.0f;
            for (size_t k=0; k<n; ++k) {
                const float *br = b+k*n;
                const float v0 = ar0[k];
                const float v1 = ar1[k];
                const float v2 = ar2[k];
                const float v3 = ar3[k];
                const float w0 = br[j+0];
                const float w1 = br[j+1];
                const float w2 = br[j+2];
                const float w3 = br[j+3];
                s00 += v0*w0;
                s01 += v0*w1;
                s02 += v0*w2;
                s03 += v0*w3;
                s10 += v1*w0;
                s11 += v1*w1;
                s12 += v1*w2;
                s13 += v1*w3;
                s20 += v2*w0;
                s21 += v2*w1;
                s22 += v2*w2;
                s23 += v2*w3;
                s30 += v3*w0;
                s31 += v3*w1;
                s32 += v3*w2;
                s33 += v3*w3;
            }
            c[(i+0)*n+j+0] = s00;
            c[(i+0)*n+j+1] = s01;
            c[(i+0)*n+j+2] = s02;
            c[(i+0)*n+j+3] = s03;
            c[(i+1)*n+j+0] = s10;
            c[(i+1)*n+j+1] = s11;
            c[(i+1)*n+j+2] = s12;
            c[(i+1)*n+j+3] = s13;
            c[(i+2)*n+j+0] = s20;
            c[(i+2)*n+j+1] = s21;
            c[(i+2)*n+j+2] = s22;
            c[(i+2)*n+j+3] = s23;
            c[(i+3)*n+j+0] = s30;
            c[(i+3)*n+j+1] = s31;
            c[(i+3)*n+j+2] = s32;
            c[(i+3)*n+j+3] = s33;
        }
    }
    for (size_t i=0; i<full; ++i) {
        const float *ar=a+i*n;
        float *cr=c+i*n;
        for (size_t j=full; j<n; ++j) {
            float sum=0.0f;
            for (size_t k=0; k<n; ++k) sum+=ar[k]*b[k*n+j];
            cr[j]=sum;
        }
    }
    for (size_t i=full; i<n; ++i) {
        const float *ar=a+i*n;
        float *cr=c+i*n;
        for (size_t k=0; k<n; ++k) {
            const float *br=b+k*n;
            const float aik=ar[k];
            for (size_t j=0; j<n; ++j) cr[j]+=aik*br[j];
        }
    }
}

// Register-block one output row and four output columns.
// Uses direct row pointers, matching Goose's four-element B sub-slice.
static void gemm_micro1x4(const float *a, const float *b, float *c, size_t n) {
    for (size_t pos=0; pos<n*n; ++pos) c[pos] = 0.0f;
    const size_t full=(n/4)*4;
    for (size_t i=0; i<n; ++i) {
        const float *ar=a+i*n;
        float *cr=c+i*n;
        for (size_t bj=0; bj<n/4; ++bj) {
            const size_t j=bj*4;
            float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
            for (size_t k=0; k<n; ++k) {
                const float *bk=b+k*n+j;
                const float value=ar[k];
                s0+=value*bk[0];
                s1+=value*bk[1];
                s2+=value*bk[2];
                s3+=value*bk[3];
            }
            cr[j]=s0; cr[j+1]=s1; cr[j+2]=s2; cr[j+3]=s3;
        }
        for (size_t j=full; j<n; ++j) {
            float sum=0.0f;
            for (size_t k=0; k<n; ++k) sum+=ar[k]*b[k*n+j];
            cr[j]=sum;
        }
    }
}

static void perform(const float *a, const float *b, float *c, size_t n,
                    const char *variant) {
    if (strcmp(variant, "ijk") == 0) gemm_ijk(a, b, c, n);
    else if (strcmp(variant, "ikj") == 0) gemm_ikj(a, b, c, n);
    else if (strcmp(variant, "blocked32") == 0) gemm_blocked32(a, b, c, n);
    else if (strcmp(variant, "rows") == 0) gemm_rows(a, b, c, n);
    else if (strcmp(variant, "micro4") == 0) gemm_micro4(a, b, c, n);
    else gemm_micro1x4(a, b, c, n);
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
         strcmp(argv[1], "blocked32") && strcmp(argv[1], "rows") && strcmp(argv[1], "micro4") && strcmp(argv[1], "micro1x4"))) {
        fprintf(stderr, "usage: gemm_study <ijk|ikj|blocked32|rows|micro4|micro1x4> <n> <repetitions>\n");
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
