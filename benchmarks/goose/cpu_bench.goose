// Matched numerical kernels for the Deep Goose CPU baseline.
// Usage: ./cpu_bench <add|relu|dot|matmul|matmul_ikj> <n> <iterations>
// A complete result line has the form RESULT,goose,operation,n,iterations,ns,checksum.
// Allocation, deterministic input construction, warmup and checksum are not timed.
import std;
import os;

fn vector_add(a: const f32[:], b: const f32[:], out: f32[:]) {
    for i in 0..a.len {
        out[i] = a[i] + b[i];
    }
}

fn vector_relu(a: const f32[:], b: const f32[:], out: f32[:]) {
    for i in 0..a.len {
        let x = a[i] + b[i];
        out[i] = if x > 0.0 { x } else { 0.0 };
    }
}

fn vector_dot(a: const f32[:], b: const f32[:]) -> f32 {
    var sum: f32 = 0.0;
    for i in 0..a.len {
        sum += a[i] * b[i];
    }
    return sum;
}

fn matrix_multiply(a: const f32[:], b: const f32[:], out: f32[:], n: i64) {
    for i in 0..n {
        for j in 0..n {
            var sum: f32 = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// Cache-friendlier row-major GEMM. Summation order per C[i,j] remains k ascending.
fn matrix_multiply_ikj(a: const f32[:], b: const f32[:], out: f32[:], n: i64) {
    for pos in 0..out.len {
        out[pos] = 0.0;
    }
    for i in 0..n {
        for k in 0..n {
            let v = a[i * n + k];
            for j in 0..n {
                out[i * n + j] += v * b[k * n + j];
            }
        }
    }
}

fn main() {
    let argv = args();
    guard argv.len == 4 else {
        print("usage: cpu_bench <add|relu|dot|matmul> <n> <iterations>");
        exit(2);
    }
    let op = argv[1];
    let n, valid_n = parse_int(argv[2]);
    let iterations, valid_iterations = parse_int(argv[3]);
    guard valid_n && valid_iterations && n > 0 && iterations > 0 && n <= 1048576 else {
        print("invalid benchmark dimensions / iterations");
        exit(2);
    }
    guard op == "add" || op == "relu" || op == "dot" || (op == "matmul" || op == "matmul_ikj") else {
        print("unknown operation");
        exit(2);
    }

    // Matmul is intentionally square in the initial matched benchmark.
    let count = if (op == "matmul" || op == "matmul_ikj") { n * n } else { n };
    var a: f32[>..] = [];
    var b: f32[>..] = [];
    var out: f32[>..] = [];
    for i in 0..count {
        a.push(((i % 257) as f32 - 128.0) * 0.0078125);
        b.push(((i % 199) as f32 - 99.0) * 0.015625);
        out.push(0.0);
    }

    // Warm up the same code path without timing it.
    var last_dot: f32 = 0.0;
    if op == "add" {
        vector_add(a, b, out);
    } else if op == "relu" {
        vector_relu(a, b, out);
    } else if op == "dot" {
        last_dot = vector_dot(a, b);
    } else {
        if op == "matmul_ikj" {
            matrix_multiply_ikj(a, b, out, n);
        } else {
            matrix_multiply(a, b, out, n);
        }
    }

    let before = clock_ns();
    for iteration in 0..iterations {
        if op == "add" {
            vector_add(a, b, out);
        } else if op == "relu" {
            vector_relu(a, b, out);
        } else if op == "dot" {
            last_dot = vector_dot(a, b);
        } else {
            if op == "matmul_ikj" {
            matrix_multiply_ikj(a, b, out, n);
        } else {
            matrix_multiply(a, b, out, n);
        }
        }
    }
    let elapsed_ns = clock_ns() - before;

    var checksum: f64 = 0.0;
    if op == "dot" {
        checksum = last_dot as f64;
    } else {
        for value in out {
            checksum += value as f64;
        }
    }
    print("RESULT,goose,", op, ",", n, ",", iterations, ",", elapsed_ns, ",", checksum);
}
