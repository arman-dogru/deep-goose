// BLAS-backed GEMM for FFI-overhead study; shapes and input data match cpu_bench.goose.
// Run ./blas_bench matmul <n> <iterations> after building with OpenBLAS installed.
import blas from "../../backends/blas";
import std;
import os;

fn main() {
    let argv = args();
    guard argv.len == 4 && argv[1] == "matmul" else {
        print("usage: blas_bench matmul <n> <iterations>");
        exit(2);
    }
    let n, ok_n = parse_int(argv[2]);
    let iterations, ok_i = parse_int(argv[3]);
    guard ok_n && ok_i && n > 0 && n <= 1024 && iterations > 0 else {
        print("invalid shape or iteration count");
        exit(2);
    }
    let count = n * n;
    var a: f32[>..] = [];
    var b: f32[>..] = [];
    var c: f32[>..] = [];
    for i in 0..count {
        a.push(((i % 257) as f32 - 128.0) * 0.0078125);
        b.push(((i % 199) as f32 - 99.0) * 0.015625);
        c.push(0.0);
    }

    blas::matmul_f32_into(a, b, c, n, n, n);
    let start = clock_ns();
    for iteration in 0..iterations {
        blas::matmul_f32_into(a, b, c, n, n, n);
    }
    let elapsed = clock_ns() - start;
    var checksum: f64 = 0.0;
    for v in c { checksum += v as f64; }
    print("RESULT,goose_blas,matmul,", n, ",", iterations, ",", elapsed, ",", checksum);
}
