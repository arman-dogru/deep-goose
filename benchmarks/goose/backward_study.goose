// Dense Y = XW+b backward comparison in Goose v2026.3.
// Computes dW = X^T dY, db = sum(dY, rows), dX = dY W^T.
// Variants: naive; rows_dw (row-sliced weight/bias gradients);
// rows_all (row-sliced weight/bias and row-sliced input gradients).
// Every gradient element is compared against independent f64 references
// OUTSIDE timing. All output buffers are caller-owned and non-aliasing.
// Usage: backward_study <naive|rows_dw|rows_all> B K N repetitions
import std;
import os;

fn backward_naive(x: const f32[:], w: const f32[:], dy: const f32[:],
                  dx: f32[:], dw: f32[:], db: f32[:],
                  batch: i64, inner: i64, outer: i64) {
    for col in 0..outer {
        var s: f32 = 0.0;
        for row in 0..batch { s += dy[row * outer + col]; }
        db[col] = s;
    }
    for k in 0..inner {
        for col in 0..outer {
            var s: f32 = 0.0;
            for row in 0..batch {
                s += x[row * inner + k] * dy[row * outer + col];
            }
            dw[k * outer + col] = s;
        }
    }
    for row in 0..batch {
        for k in 0..inner {
            var s: f32 = 0.0;
            for col in 0..outer {
                s += dy[row * outer + col] * w[k * outer + col];
            }
            dx[row * inner + k] = s;
        }
    }
}

fn backward_dw_rows(x: const f32[:], dy: const f32[:],
                    dw: f32[:], db: f32[:],
                    batch: i64, inner: i64, outer: i64) {
    for col in 0..outer { db[col] = 0.0; }
    for idx in 0..dw.len { dw[idx] = 0.0; }
    for row in 0..batch {
        let xr = x[row * inner..(row + 1) * inner];
        let dyr = dy[row * outer..(row + 1) * outer];
        for col in 0..outer { db[col] += dyr[col]; }
        for k in 0..inner {
            let dwr = dw[k * outer..(k + 1) * outer];
            let xv = xr[k];
            for col in 0..outer {
                dwr[col] += xv * dyr[col];
            }
        }
    }
}

fn backward_dx_naive(w: const f32[:], dy: const f32[:], dx: f32[:],
                     batch: i64, inner: i64, outer: i64) {
    for row in 0..batch {
        for k in 0..inner {
            var s: f32 = 0.0;
            for col in 0..outer {
                s += dy[row * outer + col] * w[k * outer + col];
            }
            dx[row * inner + k] = s;
        }
    }
}

fn backward_dx_rows(w: const f32[:], dy: const f32[:], dx: f32[:],
                    batch: i64, inner: i64, outer: i64) {
    for row in 0..batch {
        let dyr = dy[row * outer..(row + 1) * outer];
        let dxr = dx[row * inner..(row + 1) * inner];
        for k in 0..inner {
            let wr = w[k * outer..(k + 1) * outer];
            var s: f32 = 0.0;
            for col in 0..outer {
                s += dyr[col] * wr[col];
            }
            dxr[k] = s;
        }
    }
}

fn perform(variant: const u8[:],
           x: const f32[:], w: const f32[:], dy: const f32[:],
           dx: f32[:], dw: f32[:], db: f32[:],
           batch: i64, inner: i64, outer: i64) {
    if variant == "naive" {
        backward_naive(x,w,dy,dx,dw,db,batch,inner,outer);
    } else {
        backward_dw_rows(x,dy,dw,db,batch,inner,outer);
        if variant == "rows_dw" {
            backward_dx_naive(w,dy,dx,batch,inner,outer);
        } else {
            backward_dx_rows(w,dy,dx,batch,inner,outer);
        }
    }
}

fn main() {
    let argv = args();
    guard argv.len == 6 else {
        print("usage: backward_study <naive|rows_dw|rows_all> B K N repetitions");
        exit(2);
    }
    let variant = argv[1];
    let batch, okb = parse_int(argv[2]);
    let inner, okk = parse_int(argv[3]);
    let outer, okn = parse_int(argv[4]);
    let iterations, oki = parse_int(argv[5]);
    guard (variant == "naive" || variant == "rows_dw" || variant == "rows_all") &&
          okb && okk && okn && oki &&
          batch >= 1 && inner >= 1 && outer >= 1 &&
          batch <= 512 && inner <= 512 && outer <= 512 &&
          iterations >= 1 && iterations <= 10000 else {
        print("invalid backward benchmark arguments");
        exit(2);
    }
    var x: f32[>..] = [];
    var w: f32[>..] = [];
    var dy: f32[>..] = [];
    var dx: f32[>..] = [];
    var dw: f32[>..] = [];
    var db: f32[>..] = [];
    for index in 0..batch * inner {
        x.push((((index * 13) % 211) as f32 - 105.0) * 0.00317);
        dx.push(0.0);
    }
    for index in 0..inner * outer {
        w.push((((index * 23) % 191) as f32 - 95.0) * 0.00279);
        dw.push(0.0);
    }
    for index in 0..batch * outer {
        dy.push((((index * 17) % 163) as f32 - 81.0) * 0.00413);
    }
    for col in 0..outer { db.push(0.0); }

    perform(variant,x,w,dy,dx,dw,db,batch,inner,outer);
    let start = clock_ns();
    for iteration in 0..iterations {
        perform(variant,x,w,dy,dx,dw,db,batch,inner,outer);
    }
    let elapsed_ns = clock_ns() - start;

    // Independent f64 references for all three Jacobian-vector products.
    var max_error: f64 = 0.0;
    var checksum: f64 = 0.0;
    for col in 0..outer {
        var expected: f64 = 0.0;
        for row in 0..batch {
            expected += dy[row * outer + col] as f64;
        }
        let difference = (db[col] as f64) - expected;
        let error = if difference >= 0.0 { difference } else { -difference };
        let magnitude = if expected >= 0.0 { expected } else { -expected };
        assert(error <= 0.0005 + 0.0001 * magnitude);
        if error > max_error { max_error = error; }
        checksum += db[col] as f64;
    }
    for k in 0..inner {
        for col in 0..outer {
            var expected: f64 = 0.0;
            for row in 0..batch {
                expected += (x[row * inner + k] as f64) *
                            (dy[row * outer + col] as f64);
            }
            let difference = (dw[k * outer + col] as f64) - expected;
            let error = if difference >= 0.0 { difference } else { -difference };
            let magnitude = if expected >= 0.0 { expected } else { -expected };
            assert(error <= 0.0005 + 0.0001 * magnitude);
            if error > max_error { max_error = error; }
            checksum += dw[k * outer + col] as f64;
        }
    }
    for row in 0..batch {
        for k in 0..inner {
            var expected: f64 = 0.0;
            for col in 0..outer {
                expected += (dy[row * outer + col] as f64) *
                            (w[k * outer + col] as f64);
            }
            let difference = (dx[row * inner + k] as f64) - expected;
            let error = if difference >= 0.0 { difference } else { -difference };
            let magnitude = if expected >= 0.0 { expected } else { -expected };
            assert(error <= 0.0005 + 0.0001 * magnitude);
            if error > max_error { max_error = error; }
            checksum += dx[row * inner + k] as f64;
        }
    }
    print("BACKWARD_RESULT,goose,",variant,",",batch,",",inner,",",
          outer,",",iterations,",",elapsed_ns,",",max_error,",",checksum);
}
