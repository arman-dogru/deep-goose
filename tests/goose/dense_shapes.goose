// Elementwise numerical regression for optimized public dense forward.
// Checks fused bias and every output against an independent f64 reference.
import deep_goose from "../../src";

fn verify(batch: i64, inner: i64, outer: i64) {
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
    for index in 0..batch * outer { y.push(-100.0); }

    deep_goose::dense_forward_f32_into(x, w, bias, y, batch, inner, outer);
    for i in 0..batch {
        for j in 0..outer {
            var reference = bias[j] as f64;
            for k in 0..inner {
                reference += (x[i * inner + k] as f64) *
                             (w[k * outer + j] as f64);
            }
            let diff = (y[i * outer + j] as f64) - reference;
            let abs_error = if diff >= 0.0 { diff } else { -diff };
            let abs_ref = if reference >= 0.0 { reference } else { -reference };
            assert(abs_error <= 0.0005 + 0.0001 * abs_ref);
        }
    }

    // Reusing the output buffer must not accumulate values across calls.
    deep_goose::dense_forward_f32_into(x, w, bias, y, batch, inner, outer);
    for i in 0..batch {
        for j in 0..outer {
            var reference = bias[j] as f64;
            for k in 0..inner {
                reference += (x[i * inner + k] as f64) *
                             (w[k * outer + j] as f64);
            }
            let diff = (y[i * outer + j] as f64) - reference;
            let abs_error = if diff >= 0.0 { diff } else { -diff };
            let abs_ref = if reference >= 0.0 { reference } else { -reference };
            assert(abs_error <= 0.0005 + 0.0001 * abs_ref);
        }
    }
}

fn main() {
    verify(1, 1, 1);
    verify(1, 128, 128);
    verify(7, 65, 67);
    verify(32, 64, 64);
    verify(64, 128, 128);
    print("PASS row-sliced dense forward for irregular, singleton, and batch shapes");
}
