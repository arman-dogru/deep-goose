// Deep Goose: controlled row-major f32 GEMM loop-order / cache-tiling study.
// CLI: ./gemm_study <ijk|ikj|blocked32|rows|micro4|micro1x4> <n> <repetitions>
// Outputs GEMM_RESULT,goose,variant,n,repetitions,elapsed_ns,max_abs_error,checksum
// Inputs and result validation are outside the timed region.
// Every output element is checked against a separate f64 reference.
import std;
import os;

fn gemm_ijk(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for i in 0..n {
        for j in 0..n {
            var acc: f32 = 0.0;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn gemm_ikj(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for pos in 0..c.len { c[pos] = 0.0; }
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}

// 32x32x32 tiling, with explicit tail handling for dimensions not
// divisible by 32. Preserves k-ascending accumulation per C[i,j].
fn gemm_blocked32(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for pos in 0..c.len { c[pos] = 0.0; }
    let tiles = (n + 31) / 32;
    for ti in 0..tiles {
        let i0 = ti * 32;
        let i1 = if i0 + 32 < n { i0 + 32 } else { n };
        for tk in 0..tiles {
            let k0 = tk * 32;
            let k1 = if k0 + 32 < n { k0 + 32 } else { n };
            for tj in 0..tiles {
                let j0 = tj * 32;
                let j1 = if j0 + 32 < n { j0 + 32 } else { n };
                for i in i0..i1 {
                    for k in k0..k1 {
                        let aik = a[i * n + k];
                        for j in j0..j1 {
                            c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// A row-slice formulation: establish row spans before the inner j loop.
// The intent is to let Goose's BCE hoist range checks to slice creation,
// rather than rechecking indexed matrix arithmetic at every iteration.
fn gemm_rows(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for pos in 0..c.len { c[pos] = 0.0; }
    for i in 0..n {
        let a_row = a[i * n..(i + 1) * n];
        let c_row = c[i * n..(i + 1) * n];
        for k in 0..n {
            let b_row = b[k * n..(k + 1) * n];
            let aik = a_row[k];
            for j in 0..n {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

// 4x4 register-blocked GEMM: accumulate 16 output elements in scalar
// registers across the complete k dimension and store each output once.
// Zeroing is timed, as in the row-sliced kernel. Tail rows/columns fall
// back to the tested row-slice traversal with identical k accumulation.
fn gemm_micro4(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for pos in 0..c.len { c[pos] = 0.0; }
    let tiles = n / 4;
    let full = tiles * 4;
    for bi in 0..tiles {
        let i = bi * 4;
        let ar0 = a[(i + 0) * n..(i + 0 + 1) * n];
        let ar1 = a[(i + 1) * n..(i + 1 + 1) * n];
        let ar2 = a[(i + 2) * n..(i + 2 + 1) * n];
        let ar3 = a[(i + 3) * n..(i + 3 + 1) * n];
        for bj in 0..tiles {
            let j = bj * 4;
            var s00: f32 = 0.0;
            var s01: f32 = 0.0;
            var s02: f32 = 0.0;
            var s03: f32 = 0.0;
            var s10: f32 = 0.0;
            var s11: f32 = 0.0;
            var s12: f32 = 0.0;
            var s13: f32 = 0.0;
            var s20: f32 = 0.0;
            var s21: f32 = 0.0;
            var s22: f32 = 0.0;
            var s23: f32 = 0.0;
            var s30: f32 = 0.0;
            var s31: f32 = 0.0;
            var s32: f32 = 0.0;
            var s33: f32 = 0.0;
            for k in 0..n {
                let br = b[k * n..(k + 1) * n];
                let v0 = ar0[k];
                let v1 = ar1[k];
                let v2 = ar2[k];
                let v3 = ar3[k];
                let w0 = br[j + 0];
                let w1 = br[j + 1];
                let w2 = br[j + 2];
                let w3 = br[j + 3];
                s00 += v0 * w0;
                s01 += v0 * w1;
                s02 += v0 * w2;
                s03 += v0 * w3;
                s10 += v1 * w0;
                s11 += v1 * w1;
                s12 += v1 * w2;
                s13 += v1 * w3;
                s20 += v2 * w0;
                s21 += v2 * w1;
                s22 += v2 * w2;
                s23 += v2 * w3;
                s30 += v3 * w0;
                s31 += v3 * w1;
                s32 += v3 * w2;
                s33 += v3 * w3;
            }
            c[(i + 0) * n + j + 0] = s00;
            c[(i + 0) * n + j + 1] = s01;
            c[(i + 0) * n + j + 2] = s02;
            c[(i + 0) * n + j + 3] = s03;
            c[(i + 1) * n + j + 0] = s10;
            c[(i + 1) * n + j + 1] = s11;
            c[(i + 1) * n + j + 2] = s12;
            c[(i + 1) * n + j + 3] = s13;
            c[(i + 2) * n + j + 0] = s20;
            c[(i + 2) * n + j + 1] = s21;
            c[(i + 2) * n + j + 2] = s22;
            c[(i + 2) * n + j + 3] = s23;
            c[(i + 3) * n + j + 0] = s30;
            c[(i + 3) * n + j + 1] = s31;
            c[(i + 3) * n + j + 2] = s32;
            c[(i + 3) * n + j + 3] = s33;
        }
    }
    for i in 0..full {
        let ar = a[i * n..(i + 1) * n];
        let cr = c[i * n..(i + 1) * n];
        for j in full..n {
            var sum: f32 = 0.0;
            for k in 0..n {
                let br = b[k * n..(k + 1) * n];
                sum += ar[k] * br[j];
            }
            cr[j] = sum;
        }
    }
    for i in full..n {
        let ar = a[i * n..(i + 1) * n];
        let cr = c[i * n..(i + 1) * n];
        for k in 0..n {
            let br = b[k * n..(k + 1) * n];
            let aik = ar[k];
            for j in 0..n {
                cr[j] += aik * br[j];
            }
        }
    }
}

// One row and four columns per tile. A four-element B slice is created
// before its scalar loads so range validation can be hoisted to the slice.
fn gemm_micro1x4(a: const f32[:], b: const f32[:], c: f32[:], n: i64) {
    for pos in 0..c.len { c[pos] = 0.0; }
    let full = (n / 4) * 4;
    for i in 0..n {
        let ar = a[i * n..(i + 1) * n];
        let cr = c[i * n..(i + 1) * n];
        for bj in 0..(n / 4) {
            let j = bj * 4;
            var s0: f32 = 0.0;
            var s1: f32 = 0.0;
            var s2: f32 = 0.0;
            var s3: f32 = 0.0;
            for k in 0..n {
                let bk = b[k * n + j..k * n + j + 4];
                let value = ar[k];
                s0 += value * bk[0];
                s1 += value * bk[1];
                s2 += value * bk[2];
                s3 += value * bk[3];
            }
            cr[j] = s0;
            cr[j + 1] = s1;
            cr[j + 2] = s2;
            cr[j + 3] = s3;
        }
        for j in full..n {
            var sum: f32 = 0.0;
            for k in 0..n {
                sum += ar[k] * b[k * n + j];
            }
            cr[j] = sum;
        }
    }
}

fn perform(a: const f32[:], b: const f32[:], c: f32[:], n: i64,
           variant: const u8[:]) {
    if variant == "ijk" {
        gemm_ijk(a, b, c, n);
    } else if variant == "ikj" {
        gemm_ikj(a, b, c, n);
    } else if variant == "blocked32" {
        gemm_blocked32(a, b, c, n);
    } else if variant == "rows" {
        gemm_rows(a, b, c, n);
    } else if variant == "micro4" {
        gemm_micro4(a, b, c, n);
    } else {
        gemm_micro1x4(a, b, c, n);
    }
}

fn main() {
    let argv = args();
    guard argv.len == 4 else {
        print("usage: gemm_study <ijk|ikj|blocked32> <n> <repetitions>");
        exit(2);
    }
    let variant = argv[1];
    let n, ok_n = parse_int(argv[2]);
    let repetitions, ok_reps = parse_int(argv[3]);
    guard (variant == "ijk" || variant == "ikj" || variant == "blocked32" || variant == "rows" || variant == "micro4" || variant == "micro1x4") &&
          ok_n && ok_reps && n >= 1 && n <= 256 &&
          repetitions > 0 && repetitions <= 10000 else {
        print("invalid GEMM study arguments");
        exit(2);
    };
    let count = n * n;
    var a: f32[>..] = [];
    var b: f32[>..] = [];
    var c: f32[>..] = [];
    for i in 0..count {
        a.push(((i % 257) as f32 - 128.0) * 0.00713);
        b.push(((i % 199) as f32 - 99.0) * 0.01517);
        c.push(0.0);
    }

    perform(a, b, c, n, variant); // warm-up and compilation-independent setup
    let start = clock_ns();
    for repetition in 0..repetitions {
        perform(a, b, c, n, variant);
    }
    let elapsed_ns = clock_ns() - start;

    // Reference is evaluated in f64 with a different accumulation type.
    // Validates every element, including tile tails, not just a checksum.
    var max_abs_error: f64 = 0.0;
    var checksum: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            var expected: f64 = 0.0;
            for k in 0..n {
                expected += (a[i * n + k] as f64) *
                            (b[k * n + j] as f64);
            }
            let actual = c[i * n + j] as f64;
            let difference = actual - expected;
            let error = if difference >= 0.0 { difference } else { -difference };
            let magnitude = if expected >= 0.0 { expected } else { -expected };
            assert(error <= 0.0005 + 0.0001 * magnitude);
            if error > max_abs_error { max_abs_error = error; }
            checksum += actual;
        }
    }
    print("GEMM_RESULT,goose,", variant, ",", n, ",", repetitions, ",",
          elapsed_ns, ",", max_abs_error, ",", checksum);
}
