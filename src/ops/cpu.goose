// Dependency-free CPU numerical operations and a simple caller-owned 2D tensor.
// All kernels work on contiguous row-major f32 storage.
namespace deep_goose;

struct Tensor2D {
    rows: i64,
    cols: i64,
    data: f32[>..]
}

fn zeros2d(rows: i64, cols: i64) -> Tensor2D {
    // This first implementation explicitly bounds tensor sizes and prevents overflow.
    assert(rows >= 0 && cols >= 0 && rows <= 1048576 && cols <= 1048576);
    assert(cols == 0 || rows <= 1048576 / cols);
    var result = Tensor2D { rows: rows, cols: cols, data: [] };
    for i in 0..rows * cols {
        result.data.push(0.0);
    }
    return result;
}

fn fill_f32(out: f32[:], value: f32) {
    for i in 0..out.len {
        out[i] = value;
    }
}

fn add_f32_into(a: const f32[:], b: const f32[:], out: f32[:]) {
    assert(a.len == b.len && b.len == out.len);
    for i in 0..out.len {
        out[i] = a[i] + b[i];
    }
}

fn relu_f32_into(input: const f32[:], out: f32[:]) {
    assert(input.len == out.len);
    for i in 0..out.len {
        let v = input[i];
        out[i] = if v > 0.0 { v } else { 0.0 };
    }
}

fn relu_backward_f32_into(pre_activation: const f32[:], grad_y: const f32[:], grad_x: f32[:]) {
    assert(pre_activation.len == grad_y.len && grad_y.len == grad_x.len);
    for i in 0..grad_x.len {
        grad_x[i] = if pre_activation[i] > 0.0 { grad_y[i] } else { 0.0 };
    }
}

fn dot_f32(a: const f32[:], b: const f32[:]) -> f32 {
    assert(a.len == b.len);
    var result: f32 = 0.0;
    for i in 0..a.len {
        result += a[i] * b[i];
    }
    return result;
}

fn matmul_f32_into(a: const f32[:], b: const f32[:], out: f32[:],
                   m: i64, k: i64, n: i64) {
    assert(m >= 0 && k >= 0 && n >= 0);
    assert(m <= 1048576 && k <= 1048576 && n <= 1048576);
    assert(m * k == a.len && k * n == b.len && m * n == out.len);
    for row in 0..m {
        for col in 0..n {
            var sum: f32 = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

fn dense_forward_f32_into(x: const f32[:], w: const f32[:], bias: const f32[:],
                          y: f32[:], batch: i64, in_features: i64, out_features: i64) {
    assert(batch > 0 && in_features > 0 && out_features > 0);
    assert(batch <= 1048576 && in_features <= 1048576 && out_features <= 1048576);
    assert(x.len == batch * in_features);
    assert(w.len == in_features * out_features);
    assert(bias.len == out_features && y.len == batch * out_features);
    for row in 0..batch {
        for col in 0..out_features {
            var sum = bias[col];
            for k in 0..in_features {
                sum += x[row * in_features + k] * w[k * out_features + col];
            }
            y[row * out_features + col] = sum;
        }
    }
}

// Backward of Y=XW+b; gradients are sums over the batch (not averages).
fn dense_backward_f32_into(x: const f32[:], w: const f32[:], grad_y: const f32[:],
                           grad_x: f32[:], grad_w: f32[:], grad_b: f32[:],
                           batch: i64, in_features: i64, out_features: i64) {
    assert(batch > 0 && in_features > 0 && out_features > 0);
    assert(batch <= 1048576 && in_features <= 1048576 && out_features <= 1048576);
    assert(x.len == batch * in_features && w.len == in_features * out_features);
    assert(grad_y.len == batch * out_features);
    assert(grad_x.len == x.len && grad_w.len == w.len && grad_b.len == out_features);
    for col in 0..out_features {
        var sum: f32 = 0.0;
        for row in 0..batch {
            sum += grad_y[row * out_features + col];
        }
        grad_b[col] = sum;
    }
    for k in 0..in_features {
        for col in 0..out_features {
            var sum: f32 = 0.0;
            for row in 0..batch {
                sum += x[row * in_features + k] * grad_y[row * out_features + col];
            }
            grad_w[k * out_features + col] = sum;
        }
    }
    for row in 0..batch {
        for k in 0..in_features {
            var sum: f32 = 0.0;
            for col in 0..out_features {
                sum += grad_y[row * out_features + col] * w[k * out_features + col];
            }
            grad_x[row * in_features + k] = sum;
        }
    }
}

fn mse_mean_f32(y: const f32[:], target: const f32[:]) -> f32 {
    assert(y.len == target.len && y.len > 0);
    var result: f32 = 0.0;
    for i in 0..y.len {
        let diff = y[i] - target[i];
        result += diff * diff;
    }
    return result / (y.len as f32);
}

fn mse_backward_f32_into(y: const f32[:], target: const f32[:], grad_y: f32[:]) {
    assert(y.len == target.len && target.len == grad_y.len && y.len > 0);
    let scale = 2.0 / (y.len as f32);
    for i in 0..y.len {
        grad_y[i] = scale * (y[i] - target[i]);
    }
}

fn sgd_step_f32(parameters: f32[:], gradients: const f32[:], learning_rate: f32) {
    assert(parameters.len == gradients.len);
    assert(learning_rate > 0.0);
    for i in 0..parameters.len {
        parameters[i] -= learning_rate * gradients[i];
    }
}
