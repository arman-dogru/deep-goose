// Numerically stable row-wise softmax, log-softmax and classification loss.
// Contiguous row-major f32 logits: [rows, classes].
// Loss/gradient use the mean over rows (NOT over all elements).
import math;
namespace deep_goose;

fn softmax_rows_f32_into(logits: const f32[:], output: f32[:],
                         rows: i64, classes: i64) {
    assert(rows > 0 && classes > 0 && rows <= 1048576 && classes <= 1048576);
    assert(rows <= 1048576 / classes);
    assert(logits.len == rows * classes && output.len == logits.len);
    for row in 0..rows {
        let offset = row * classes;
        var largest = logits[offset];
        for c in 1..classes {
            if logits[offset + c] > largest {
                largest = logits[offset + c];
            }
        }
        var denominator: f32 = 0.0;
        for c in 0..classes {
            let value = exp(logits[offset + c] - largest);
            output[offset + c] = value;
            denominator += value;
        }
        for c in 0..classes {
            output[offset + c] /= denominator;
        }
    }
}

fn log_softmax_rows_f32_into(logits: const f32[:], output: f32[:],
                             rows: i64, classes: i64) {
    assert(rows > 0 && classes > 0 && rows <= 1048576 && classes <= 1048576);
    assert(rows <= 1048576 / classes);
    assert(logits.len == rows * classes && output.len == logits.len);
    for row in 0..rows {
        let offset = row * classes;
        var largest = logits[offset];
        for c in 1..classes {
            if logits[offset + c] > largest { largest = logits[offset + c]; }
        }
        var denominator: f32 = 0.0;
        for c in 0..classes {
            denominator += exp(logits[offset + c] - largest);
        }
        let log_sum_exp = largest + log(denominator);
        for c in 0..classes {
            output[offset + c] = logits[offset + c] - log_sum_exp;
        }
    }
}

fn cross_entropy_logits_mean_f32(logits: const f32[:], targets: const i64[:],
                                  rows: i64, classes: i64) -> f32 {
    assert(rows > 0 && classes > 0 && rows <= 1048576 && classes <= 1048576);
    assert(rows <= 1048576 / classes);
    assert(logits.len == rows * classes && targets.len == rows);
    var total_loss: f32 = 0.0;
    for row in 0..rows {
        let target = targets[row];
        assert(target >= 0 && target < classes);
        let offset = row * classes;
        var largest = logits[offset];
        for c in 1..classes {
            if logits[offset + c] > largest { largest = logits[offset + c]; }
        }
        var denominator: f32 = 0.0;
        for c in 0..classes {
            denominator += exp(logits[offset + c] - largest);
        }
        // Compute log-sum-exp in shifted form to avoid overflow and
        // cancellation for very large equal-valued logits.
        total_loss += log(denominator) + (largest - logits[offset + target]);
    }
    total_loss / (rows as f32)
}

fn cross_entropy_logits_backward_f32_into(logits: const f32[:],
                                            targets: const i64[:],
                                            gradient: f32[:],
                                            rows: i64, classes: i64) {
    assert(rows > 0 && classes > 0 && rows <= 1048576 && classes <= 1048576);
    assert(rows <= 1048576 / classes);
    assert(logits.len == rows * classes && gradient.len == logits.len);
    assert(targets.len == rows);
    for row in 0..rows {
        let target = targets[row];
        assert(target >= 0 && target < classes);
        let offset = row * classes;
        var largest = logits[offset];
        for c in 1..classes {
            if logits[offset + c] > largest { largest = logits[offset + c]; }
        }
        var denominator: f32 = 0.0;
        for c in 0..classes {
            let value = exp(logits[offset + c] - largest);
            gradient[offset + c] = value;
            denominator += value;
        }
        let scale = (1.0 / denominator) / (rows as f32);
        for c in 0..classes {
            gradient[offset + c] *= scale;
        }
        gradient[offset + target] -= 1.0 / (rows as f32);
    }
}
