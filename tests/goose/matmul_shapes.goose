// Complete-element numerical tests for the optimized row-sliced CPU GEMM.
// Includes rectangular matrices, non-multiples of 32, and zero inner size.
import deep_goose from "../../src";

fn check_shape(m: i64, k: i64, n: i64) {
    var a: f32[>..] = [];
    var b: f32[>..] = [];
    var out: f32[>..] = [];
    for index in 0..m * k {
        a.push((((index * 17) % 31) as f32 - 15.0) * 0.125);
    }
    for index in 0..k * n {
        b.push((((index * 13) % 29) as f32 - 14.0) * 0.125);
    }
    for index in 0..m * n { out.push(-999.0); }
    deep_goose::matmul_f32_into(a, b, out, m, k, n);
    for i in 0..m {
        for j in 0..n {
            var expected: f64 = 0.0;
            for inner in 0..k {
                expected += (a[i * k + inner] as f64) *
                            (b[inner * n + j] as f64);
            }
            let actual = out[i * n + j] as f64;
            let delta = actual - expected;
            let abs_error = if delta >= 0.0 { delta } else { -delta };
            let abs_ref = if expected >= 0.0 { expected } else { -expected };
            assert(abs_error <= 0.0005 + 0.0001 * abs_ref);
        }
    }
}

fn main() {
    check_shape(1, 1, 1);
    check_shape(2, 3, 4);
    check_shape(7, 5, 9);
    check_shape(33, 34, 31);
    check_shape(65, 63, 67);
    check_shape(1, 0, 3);
    check_shape(0, 3, 7);
    print("PASS row-sliced CPU GEMM for rectangular/tail/zero-sized shapes");
}
