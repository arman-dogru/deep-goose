// End-to-end nonlinear MLP training on XOR in native Goose.
// 2 -> 8 (ReLU) -> 1, mean squared error, explicit backpropagation, SGD.
import deep_goose from "../../src";

fn main() {
    let inputs: f32[8] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let targets: f32[4] = [0.0, 1.0, 1.0, 0.0];

    var w1: f32[16] = [0.0; 16];
    var b1: f32[8] = [0.0; 8];
    var w2: f32[8] = [0.0; 8];
    var b2: f32[1] = [0.0];
    for i in 0..16 {
        w1[i] = (((i * 37) % 19) as f32 - 9.0) * 0.08;
    }
    for j in 0..8 {
        b1[j] = (((j * 11) % 13) as f32 - 6.0) * 0.04;
        w2[j] = (((j * 17) % 23) as f32 - 11.0) * 0.07;
    }

    var pre_activation: f32[32] = [0.0; 32];
    var hidden: f32[32] = [0.0; 32];
    var predictions: f32[4] = [0.0; 4];
    var grad_predictions: f32[4] = [0.0; 4];
    var grad_hidden: f32[32] = [0.0; 32];
    var grad_pre: f32[32] = [0.0; 32];
    var grad_inputs: f32[8] = [0.0; 8];
    var grad_w1: f32[16] = [0.0; 16];
    var grad_b1: f32[8] = [0.0; 8];
    var grad_w2: f32[8] = [0.0; 8];
    var grad_b2: f32[1] = [0.0];

    for epoch in 0..4000 {
        deep_goose::dense_forward_f32_into(
            inputs, w1, b1, pre_activation, 4, 2, 8
        );
        deep_goose::relu_f32_into(pre_activation, hidden);
        deep_goose::dense_forward_f32_into(
            hidden, w2, b2, predictions, 4, 8, 1
        );

        deep_goose::mse_backward_f32_into(predictions, targets, grad_predictions);
        deep_goose::dense_backward_f32_into(
            hidden, w2, grad_predictions, grad_hidden, grad_w2, grad_b2, 4, 8, 1
        );
        deep_goose::relu_backward_f32_into(pre_activation, grad_hidden, grad_pre);
        deep_goose::dense_backward_f32_into(
            inputs, w1, grad_pre, grad_inputs, grad_w1, grad_b1, 4, 2, 8
        );

        deep_goose::sgd_step_f32(w1, grad_w1, 0.1);
        deep_goose::sgd_step_f32(b1, grad_b1, 0.1);
        deep_goose::sgd_step_f32(w2, grad_w2, 0.1);
        deep_goose::sgd_step_f32(b2, grad_b2, 0.1);
    }

    deep_goose::dense_forward_f32_into(inputs, w1, b1, pre_activation, 4, 2, 8);
    deep_goose::relu_f32_into(pre_activation, hidden);
    deep_goose::dense_forward_f32_into(hidden, w2, b2, predictions, 4, 8, 1);
    let loss = deep_goose::mse_mean_f32(predictions, targets);
    assert(loss < 0.0001);
    print("PASS XOR MLP; MSE=", loss, "; predictions=", predictions);
}
