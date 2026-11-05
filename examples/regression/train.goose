// End-to-end CPU training experiment: 2-input linear regression with SGD.
// Uses Deep Goose's dense forward, mean squared error, backward, and SGD.
// Entire training loop is timed; dataset construction and final evaluation are not.
import deep_goose from "../../src";
import os;

fn main() {
    let batch: i64 = 256;
    let epochs: i64 = 200;
    let lr: f32 = 0.1;

    var x: f32[512] = [0.0; 512];
    var target: f32[256] = [0.0; 256];
    for row in 0..batch {
        let f0 = ((row % 17) as f32 - 8.0) / 8.0;
        let f1 = (((row * 7) % 19) as f32 - 9.0) / 9.0;
        x[row * 2] = f0;
        x[row * 2 + 1] = f1;
        target[row] = 1.5 * f0 - 2.0 * f1 + 0.25;
    }

    var w: f32[2] = [0.0; 2];
    var bias: f32[1] = [0.0; 1];
    var prediction: f32[256] = [0.0; 256];
    var grad_y: f32[256] = [0.0; 256];
    var grad_x: f32[512] = [0.0; 512];
    var grad_w: f32[2] = [0.0; 2];
    var grad_bias: f32[1] = [0.0; 1];

    let before = clock_ns();
    for epoch in 0..epochs {
        deep_goose::dense_forward_f32_into(x, w, bias, prediction, batch, 2, 1);
        deep_goose::mse_backward_f32_into(prediction, target, grad_y);
        deep_goose::dense_backward_f32_into(
            x, w, grad_y, grad_x, grad_w, grad_bias, batch, 2, 1
        );
        deep_goose::sgd_step_f32(w, grad_w, lr);
        deep_goose::sgd_step_f32(bias, grad_bias, lr);
    }
    let elapsed_ns = clock_ns() - before;

    deep_goose::dense_forward_f32_into(x, w, bias, prediction, batch, 2, 1);
    let final_loss = deep_goose::mse_mean_f32(prediction, target);
    assert(final_loss < 0.0001);
    print("TRAIN_RESULT,goose,", epochs, ",", elapsed_ns, ",",
          final_loss, ",", w[0], ",", w[1], ",", bias[0]);
}
