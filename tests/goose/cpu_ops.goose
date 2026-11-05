// Compiler and numerical regression tests for the initial Goose CPU library.
import deep_goose from "../../src";

fn near(a: f32, b: f32, tolerance: f32) -> bool {
    let difference = if a > b { a - b } else { b - a };
    difference <= tolerance
}

fn main() {
    var t = deep_goose::zeros2d(2, 3);
    assert(t.rows == 2 && t.cols == 3 && t.data.len == 6);
    t.data[4] = 9.0;
    assert(t.data[4] == 9.0);

    let x: f32[4] = [1.0, 2.0, 3.0, 4.0];
    let w: f32[4] = [1.0, 2.0, 3.0, 4.0];
    var y: f32[4] = [0.0; 4];
    deep_goose::matmul_f32_into(x, w, y, 2, 2, 2);
    assert(near(y[0], 7.0, 0.0001));
    assert(near(y[1], 10.0, 0.0001));
    assert(near(y[2], 15.0, 0.0001));
    assert(near(y[3], 22.0, 0.0001));

    let bias: f32[2] = [0.5, -0.5];
    deep_goose::dense_forward_f32_into(x, w, bias, y, 2, 2, 2);
    assert(near(y[0], 7.5, 0.0001));
    assert(near(y[1], 9.5, 0.0001));
    assert(near(y[2], 15.5, 0.0001));
    assert(near(y[3], 21.5, 0.0001));

    let grad_y: f32[4] = [1.0, 2.0, 3.0, 4.0];
    var grad_x: f32[4] = [0.0; 4];
    var grad_w: f32[4] = [0.0; 4];
    var grad_b: f32[2] = [0.0; 2];
    deep_goose::dense_backward_f32_into(x, w, grad_y, grad_x, grad_w, grad_b, 2, 2, 2);
    assert(near(grad_w[0], 10.0, 0.0001));
    assert(near(grad_w[1], 14.0, 0.0001));
    assert(near(grad_w[2], 14.0, 0.0001));
    assert(near(grad_w[3], 20.0, 0.0001));
    assert(near(grad_b[0], 4.0, 0.0001));
    assert(near(grad_b[1], 6.0, 0.0001));
    assert(near(grad_x[0], 5.0, 0.0001));
    assert(near(grad_x[1], 11.0, 0.0001));
    assert(near(grad_x[2], 11.0, 0.0001));
    assert(near(grad_x[3], 25.0, 0.0001));

    var q: f32[4] = [0.0; 4];
    let transition = deep_goose::Transition {
        state: 0, action: 1, reward: 1.0,
        next_state: 1, terminated: true, truncated: false
    };
    deep_goose::q_learning_update_f32(q, 2, 2, transition, 0.5, 0.9);
    assert(near(q[1], 0.5, 0.00001));

    var replay = deep_goose::new_replay(2);
    deep_goose::replay_push(replay, transition);
    deep_goose::replay_push(replay, transition);
    let later = deep_goose::Transition {
        state: 1, action: 0, reward: 2.0,
        next_state: 0, terminated: false, truncated: true
    };
    deep_goose::replay_push(replay, later);
    assert(replay.entries.len == 2);
    assert(deep_goose::replay_get(replay, 0).reward == 2.0);
    assert(deep_goose::replay_get(replay, 1).reward == 1.0);

    var a: f32[4] = [0.0; 4];
    deep_goose::add_f32_into(x, w, a);
    assert(a[0] == 2.0 && a[3] == 8.0);
    deep_goose::relu_f32_into(a, a);
    assert(a[0] == 2.0 && a[3] == 8.0);
    let loss = deep_goose::mse_mean_f32(y, y);
    assert(near(loss, 0.0, 0.00001));
    var gy: f32[4] = [0.0; 4];
    deep_goose::mse_backward_f32_into(y, y, gy);
    assert(gy[0] == 0.0 && gy[3] == 0.0);
    deep_goose::sgd_step_f32(a, x, 0.1);
    assert(near(a[0], 1.9, 0.00001));

    var adam_params: f32[2] = [1.0, -1.0];
    let adam_grad: f32[2] = [2.0, -2.0];
    var adam_m: f32[2] = [0.0; 2];
    var adam_v: f32[2] = [0.0; 2];
    deep_goose::adam_step_f32(adam_params, adam_grad, adam_m, adam_v,
                              0.001, 0.9, 0.999, 0.1, 0.001, 0.00000001);
    assert(near(adam_m[0], 0.2, 0.00001));
    assert(near(adam_v[0], 0.004, 0.00001));
    assert(near(adam_params[0], 0.999, 0.00001));
    assert(near(adam_params[1], -0.999, 0.00001));

    print("PASS deep_goose CPU tensor, dense forward/backward, Q-learning, replay");
}
