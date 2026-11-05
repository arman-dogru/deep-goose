// Deep Goose: controlled row-major f32 GEMM loop-order / cache-tiling study.
// CLI: ./gemm_study <ijk|ikj|blocked32> <n> <repetitions>
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

fn perform(a: const f32[:], b: const f32[:], c: f32[:], n: i64,
           variant: const u8[:]) {
    if variant == "ijk" {
        gemm_ijk(a, b, c, n);
    } else if variant == "ikj" {
        gemm_ikj(a, b, c, n);
    } else {
        gemm_blocked32(a, b, c, n);
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
    guard (variant == "ijk" || variant == "ikj" || variant == "blocked32") &&
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
        a.push(((i % 257) as f32 - 128.0) * 0.0078125);
        b.push(((i % 199) as f32 - 99.0) * 0.015625);
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
