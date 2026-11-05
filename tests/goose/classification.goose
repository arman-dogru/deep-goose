// Validation of stable cross entropy, softmax, log-softmax and f32 gradients.
import deep_goose from "../../src";

fn close(actual: f32, expected: f32, tolerance: f32) -> bool {
    let error = if actual > expected { actual - expected } else { expected - actual };
    error <= tolerance
}

fn main() {
    // Large logits would overflow naive exp(logit) softmax.
    let logits: f32[6] = [1000.0, 1001.0, 999.0, -1000.0, -999.0, -1001.0];
    let targets: i64[2] = [1, 2];
    var probs: f32[6] = [0.0; 6];
    var log_probs: f32[6] = [0.0; 6];
    var gradients: f32[6] = [0.0; 6];

    deep_goose::softmax_rows_f32_into(logits, probs, 2, 3);
    deep_goose::log_softmax_rows_f32_into(logits, log_probs, 2, 3);
    let loss = deep_goose::cross_entropy_logits_mean_f32(logits, targets, 2, 3);
    deep_goose::cross_entropy_logits_backward_f32_into(logits, targets, gradients, 2, 3);

    assert(close(probs[0], 0.24472847, 0.00001));
    assert(close(probs[1], 0.66524096, 0.00001));
    assert(close(probs[2], 0.09003057, 0.00001));
    assert(close(probs[3], 0.24472847, 0.00001));
    assert(close(probs[4], 0.66524096, 0.00001));
    assert(close(probs[5], 0.09003057, 0.00001));

    assert(close(log_probs[0], -1.40760596, 0.0001));
    assert(close(log_probs[1], -0.40760596, 0.0001));
    assert(close(log_probs[2], -2.40760596, 0.0001));
    assert(close(loss, 1.40760596, 0.0001));
    assert(close(gradients[0], 0.12236424, 0.00001));
    assert(close(gradients[1], -0.16737952, 0.00001));
    assert(close(gradients[2], 0.04501529, 0.00001));
    assert(close(gradients[3], 0.12236424, 0.00001));
    assert(close(gradients[4], 0.33262048, 0.00001));
    assert(close(gradients[5], -0.45498471, 0.00001));

    // Three identical logits must yield uniform probabilities. Extreme
    // equal logits also catch cancellation in logsumexp implementations.
    let equal_logits: f32[3] = [10000.0, 10000.0, 10000.0];
    let equal_target: i64[1] = [0];
    var equal_probs: f32[3] = [0.0; 3];
    deep_goose::softmax_rows_f32_into(equal_logits, equal_probs, 1, 3);
    assert(close(equal_probs[0], 0.3333333, 0.00001));
    let equal_loss = deep_goose::cross_entropy_logits_mean_f32(
        equal_logits, equal_target, 1, 3
    );
    assert(close(equal_loss, 1.0986123, 0.0001));

    // Finite difference check on the same stable loss, using moderate logits
    // to avoid f32 precision loss at |logit| ~= 1000.
    var moderate: f32[6] = [0.3, -0.8, 1.2, -0.1, 0.7, -1.4];
    deep_goose::cross_entropy_logits_backward_f32_into(
        moderate, targets, gradients, 2, 3
    );
    let epsilon: f32 = 0.002;
    for i in 0..6 {
        let original = moderate[i];
        moderate[i] = original + epsilon;
        let positive = deep_goose::cross_entropy_logits_mean_f32(
            moderate, targets, 2, 3
        );
        moderate[i] = original - epsilon;
        let negative = deep_goose::cross_entropy_logits_mean_f32(
            moderate, targets, 2, 3
        );
        moderate[i] = original;
        let estimated = (positive - negative) / (2.0 * epsilon);
        assert(close(estimated, gradients[i], 0.0003));
    }
    print("PASS stable softmax, log-softmax, cross entropy, and numerical gradients");
}
