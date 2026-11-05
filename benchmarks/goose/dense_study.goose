// Matched Goose/C CPU dense forward benchmark.
// Computes Y = XW + b on f32 row-major tensors, with independent f64 validation.
// Usage: dense_study <naive|rows> <batch> <in_features> <out_features> <iterations>
// Result: DENSE_RESULT,goose,variant,batch,in,out,iterations,elapsed_ns,max_error,checksum
import std;
import os;

fn dense_naive(x: const f32[:], w: const f32[:], bias: const f32[:],
               y: f32[:], batch: i64, inner: i64, outer: i64) {
    for i in 0..batch {
        for j in 0..outer {
            var sum = bias[j];
            for k in 0..inner {
                sum += x[i * inner + k] * w[k * outer + j];
            }
            y[i * outer + j] = sum;
        }
    }
}

fn dense_rows(x: const f32[:], w: const f32[:], bias: const f32[:],
              y: f32[:], batch: i64, inner: i64, outer: i64) {
    for i in 0..batch {
        let xrow = x[i * inner..(i + 1) * inner];
        let yrow = y[i * outer..(i + 1) * outer];
        for j in 0..outer { yrow[j] = bias[j]; }
        for k in 0..inner {
            let wrow = w[k * outer..(k + 1) * outer];
            let value = xrow[k];
            for j in 0..outer {
                yrow[j] += value * wrow[j];
            }
        }
    }
}

fn main() {
    let argv = args();
    guard argv.len == 6 else {
        print("usage: dense_study <naive|rows> <batch> <in> <out> <iterations>");
        exit(2);
    }
    let variant = argv[1];
    let batch, good_batch = parse_int(argv[2]);
    let inner, good_inner = parse_int(argv[3]);
    let outer, good_outer = parse_int(argv[4]);
    let iterations, good_iterations = parse_int(argv[5]);
    guard (variant == "naive" || variant == "rows") &&
          good_batch && good_inner && good_outer && good_iterations &&
          batch >= 1 && inner >= 1 && outer >= 1 &&
          batch <= 1024 && inner <= 512 && outer <= 512 &&
          iterations > 0 && iterations <= 10000 else {
        print("invalid dense benchmark arguments");
        exit(2);
    }
    var x: f32[>..] = [];
    var w: f32[>..] = [];
    var bias: f32[>..] = [];
    var y: f32[>..] = [];
    for index in 0..batch * inner {
        x.push((((index * 13) % 211) as f32 - 105.0) * 0.00317);
    }
    for index in 0..inner * outer {
        w.push((((index * 23) % 191) as f32 - 95.0) * 0.00279);
    }
    for index in 0..outer {
        bias.push((((index * 7) % 53) as f32 - 26.0) * 0.00831);
    }
    for index in 0..batch * outer { y.push(0.0); }

    if variant == "naive" {
        dense_naive(x, w, bias, y, batch, inner, outer);
    } else {
        dense_rows(x, w, bias, y, batch, inner, outer);
    }
    let start = clock_ns();
    for iteration in 0..iterations {
        if variant == "naive" {
            dense_naive(x, w, bias, y, batch, inner, outer);
        } else {
            dense_rows(x, w, bias, y, batch, inner, outer);
        }
    }
    let elapsed_ns = clock_ns() - start;

    var max_error: f64 = 0.0;
    var checksum: f64 = 0.0;
    for i in 0..batch {
        for j in 0..outer {
            var reference = bias[j] as f64;
            for k in 0..inner {
                reference += (x[i * inner + k] as f64) *
                             (w[k * outer + j] as f64);
            }
            let actual = y[i * outer + j] as f64;
            let difference = actual - reference;
            let abs_error = if difference >= 0.0 { difference } else { -difference };
            let abs_expected = if reference >= 0.0 { reference } else { -reference };
            assert(abs_error <= 0.0005 + 0.0001 * abs_expected);
            if abs_error > max_error { max_error = abs_error; }
            checksum += actual;
        }
    }
    print("DENSE_RESULT,goose,", variant, ",", batch, ",", inner, ",",
          outer, ",", iterations, ",", elapsed_ns, ",", max_error, ",", checksum);
}
