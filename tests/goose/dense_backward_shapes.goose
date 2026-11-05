// Public dense backward API: complete dW/db/dX numerical coverage across shapes.
// Independent f64 references, non-dyadic f32 inputs, and repeated-call checks.
// Tests B=1, singleton and non-square shapes, and larger batched dimensions.
import deep_goose from "../../src";

fn verify(batch: i64, inner: i64, outer: i64) {
    var x: f32[>..] = [];
    var w: f32[>..] = [];
    var dy: f32[>..] = [];
    var dx: f32[>..] = [];
    var dw: f32[>..] = [];
    var db: f32[>..] = [];
    for idx in 0..batch * inner {
        x.push((((idx * 13) % 211) as f32 - 105.0) * 0.00317);
        dx.push(-77.0);
    }
    for idx in 0..inner * outer {
        w.push((((idx * 23) % 191) as f32 - 95.0) * 0.00279);
        dw.push(-77.0);
    }
    for idx in 0..batch * outer {
        dy.push((((idx * 17) % 163) as f32 - 81.0) * 0.00413);
    }
    for col in 0..outer { db.push(-77.0); }

    // Repeating the call must overwrite all gradients, rather than
    // accumulating from the caller's previous buffer content.
    for repetition in 0..2 {
        deep_goose::dense_backward_f32_into(x, w, dy, dx, dw, db,
                                           batch, inner, outer);
        for col in 0..outer {
            var expected: f64 = 0.0;
            for row in 0..batch {
                expected += dy[row * outer + col] as f64;
            }
            let diff = (db[col] as f64) - expected;
            let error = if diff >= 0.0 { diff } else { -diff };
            let magnitude = if expected >= 0.0 { expected } else { -expected };
            assert(error <= 0.0005 + 0.0001 * magnitude);
        }
        for k in 0..inner {
            for col in 0..outer {
                var expected: f64 = 0.0;
                for row in 0..batch {
                    expected += (x[row * inner + k] as f64) *
                                (dy[row * outer + col] as f64);
                }
                let diff = (dw[k * outer + col] as f64) - expected;
                let error = if diff >= 0.0 { diff } else { -diff };
                let magnitude = if expected >= 0.0 { expected } else { -expected };
                assert(error <= 0.0005 + 0.0001 * magnitude);
            }
        }
        for row in 0..batch {
            for k in 0..inner {
                var expected: f64 = 0.0;
                for col in 0..outer {
                    expected += (dy[row * outer + col] as f64) *
                                (w[k * outer + col] as f64);
                }
                let diff = (dx[row * inner + k] as f64) - expected;
                let error = if diff >= 0.0 { diff } else { -diff };
                let magnitude = if expected >= 0.0 { expected } else { -expected };
                assert(error <= 0.0005 + 0.0001 * magnitude);
            }
        }
    }
}

fn main() {
    verify(1, 1, 1);
    verify(1, 7, 9);
    verify(7, 65, 67);
    verify(32, 64, 64);
    verify(64, 128, 128);
    print("PASS dense backward dW/db/dX elementwise, irregular shapes and repeated calls");
}
