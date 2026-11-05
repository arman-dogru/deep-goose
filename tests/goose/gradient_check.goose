// Finite-difference checks for dense-forward / MSE backward.
// Analytical gradients must agree with central differences at selected weights.
import deep_goose from "../../src";

fn loss_for(x: const f32[:], w: const f32[:], b: const f32[:],
            target: const f32[:]) -> f32 {
    var prediction: f32[2] = [0.0; 2];
    deep_goose::dense_forward_f32_into(x, w, b, prediction, 2, 2, 1);
    return deep_goose::mse_mean_f32(prediction, target);
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    let difference = if a > b { a - b } else { b - a };
    difference <= tolerance
}

fn main() {
    var x: f32[4] = [0.5, -1.25, 1.5, 0.75];
    let target: f32[2] = [0.25, -0.8];
    var w: f32[2] = [0.3, -0.4];
    var b: f32[1] = [0.1];
    var prediction: f32[2] = [0.0; 2];
    var dy: f32[2] = [0.0; 2];
    var dx: f32[4] = [0.0; 4];
    var dw: f32[2] = [0.0; 2];
    var db: f32[1] = [0.0];

    deep_goose::dense_forward_f32_into(x, w, b, prediction, 2, 2, 1);
    deep_goose::mse_backward_f32_into(prediction, target, dy);
    deep_goose::dense_backward_f32_into(x, w, dy, dx, dw, db, 2, 2, 1);
    let epsilon: f32 = 0.001;

    for i in 0..2 {
        let original = w[i];
        w[i] = original + epsilon;
        let positive = loss_for(x, w, b, target);
        w[i] = original - epsilon;
        let negative = loss_for(x, w, b, target);
        w[i] = original;
        let estimated = (positive - negative) / (2.0 * epsilon);
        assert(close(estimated, dw[i], 0.003));
    }
    let original = b[0];
    b[0] = original + epsilon;
    let positive = loss_for(x, w, b, target);
    b[0] = original - epsilon;
    let negative = loss_for(x, w, b, target);
    b[0] = original;
    let estimated = (positive - negative) / (2.0 * epsilon);
    assert(close(estimated, db[0], 0.003));

    // Verify dX too. The original regression test only covered dW and db.
    for i in 0..4 {
        let initial = x[i];
        x[i] = initial + epsilon;
        let positive_input = loss_for(x, w, b, target);
        x[i] = initial - epsilon;
        let negative_input = loss_for(x, w, b, target);
        x[i] = initial;
        let estimated_input = (positive_input - negative_input) / (2.0 * epsilon);
        assert(close(estimated_input, dx[i], 0.003));
    }

    print("PASS finite-difference dense input, weight and bias gradients");
}
