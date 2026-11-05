// Dependency-free Deep Q-Network example in native Goose CPU.
// Fixed-shape architecture: one-hot state[8] -> Dense(8,16) -> ReLU -> Dense(16,2).
// Environment: 8-state LineWorld; action 0 left, 1 right, state 7 terminal.
// Replay: fixed-capacity ring, sampled minibatches, separate target network.
// Learner: exact explicit dense-layer backward operations and SGD.
// This validates a full CPU DQN path, not benchmark-equivalence with PyTorch.
import deep_goose from "../../src";
import os;

fn main() {
    let batch: i64 = 32;
    let training_steps: i64 = 12000;
    let learning_rate: f32 = 0.03;
    let discount: f32 = 0.95;
    var seed: i64 = 1234567;
    var state: i64 = 0;
    var completed: i64 = 0;

    var w1: f32[128] = [0.0; 128];
    var b1: f32[16] = [0.0; 16];
    var w2: f32[32] = [0.0; 32];
    var b2: f32[2] = [0.0; 2];
    var target_w1: f32[128] = [0.0; 128];
    var target_b1: f32[16] = [0.0; 16];
    var target_w2: f32[32] = [0.0; 32];
    var target_b2: f32[2] = [0.0; 2];

    for i in 0..128 {
        w1[i] = (((i * 37) % 101) as f32 - 50.0) * 0.009;
        target_w1[i] = w1[i];
    }
    for i in 0..32 {
        w2[i] = (((i * 19) % 67) as f32 - 33.0) * 0.008;
        target_w2[i] = w2[i];
    }
    for i in 0..16 { b1[i] = 0.01; target_b1[i] = b1[i]; }

    var online_obs: f32[8] = [0.0; 8];
    var online_pre: f32[16] = [0.0; 16];
    var online_hidden: f32[16] = [0.0; 16];
    var online_q: f32[2] = [0.0; 2];

    var obs_batch: f32[256] = [0.0; 256];
    var next_batch: f32[256] = [0.0; 256];
    var selected_actions: i64[32] = [0; 32];
    var rewards: f32[32] = [0.0; 32];
    var terminals: bool[32] = [false; 32];

    var pre1: f32[512] = [0.0; 512];
    var hidden1: f32[512] = [0.0; 512];
    var q_values: f32[64] = [0.0; 64];
    var target_pre1: f32[512] = [0.0; 512];
    var target_hidden: f32[512] = [0.0; 512];
    var target_q_values: f32[64] = [0.0; 64];

    var grad_q: f32[64] = [0.0; 64];
    var grad_hidden: f32[512] = [0.0; 512];
    var grad_pre1: f32[512] = [0.0; 512];
    var grad_obs: f32[256] = [0.0; 256];
    var grad_w1: f32[128] = [0.0; 128];
    var grad_b1: f32[16] = [0.0; 16];
    var grad_w2: f32[32] = [0.0; 32];
    var grad_b2: f32[2] = [0.0; 2];

    var replay = deep_goose::new_replay(1024);
    var learner_updates: i64 = 0;
    let started = clock_ns();

    for iteration in 0..training_steps {
        deep_goose::fill_f32(online_obs, 0.0);
        online_obs[state] = 1.0;
        deep_goose::dense_forward_f32_into(
            online_obs, w1, b1, online_pre, 1, 8, 16
        );
        deep_goose::relu_f32_into(online_pre, online_hidden);
        deep_goose::dense_forward_f32_into(
            online_hidden, w2, b2, online_q, 1, 16, 2
        );

        seed = (seed * 1664525 + 1013904223) % 4294967296;
        let explore = (seed % 1000) < 150;
        let action: i64 = if explore {
            if seed % 2 == 0 { 0 } else { 1 }
        } else {
            if online_q[1] >= online_q[0] { 1 } else { 0 }
        };
        let next_state: i64 = if action == 0 {
            if state > 0 { state - 1 } else { 0 }
        } else {
            if state < 7 { state + 1 } else { 7 }
        };
        let terminal = next_state == 7;
        let transition = deep_goose::Transition {
            state: state, action: action,
            reward: if terminal { 1.0 } else { 0.0 },
            next_state: next_state, terminated: terminal, truncated: false
        };
        deep_goose::replay_push(replay, transition);
        if terminal {
            completed += 1;
            state = 0;
        } else {
            state = next_state;
        }

        if replay.entries.len < batch { continue; }

        deep_goose::fill_f32(obs_batch, 0.0);
        deep_goose::fill_f32(next_batch, 0.0);
        deep_goose::fill_f32(grad_q, 0.0);

        for row in 0..batch {
            seed = (seed * 1664525 + 1013904223) % 4294967296;
            let sample_idx = seed % replay.entries.len;
            let t = deep_goose::replay_get(replay, sample_idx);
            obs_batch[row * 8 + t.state] = 1.0;
            next_batch[row * 8 + t.next_state] = 1.0;
            selected_actions[row] = t.action;
            rewards[row] = t.reward;
            terminals[row] = t.terminated;
        }

        deep_goose::dense_forward_f32_into(
            next_batch, target_w1, target_b1, target_pre1, batch, 8, 16
        );
        deep_goose::relu_f32_into(target_pre1, target_hidden);
        deep_goose::dense_forward_f32_into(
            target_hidden, target_w2, target_b2, target_q_values, batch, 16, 2
        );
        deep_goose::dense_forward_f32_into(
            obs_batch, w1, b1, pre1, batch, 8, 16
        );
        deep_goose::relu_f32_into(pre1, hidden1);
        deep_goose::dense_forward_f32_into(
            hidden1, w2, b2, q_values, batch, 16, 2
        );

        // Gradient of sum of squared chosen-action TD errors / batch.
        for row in 0..batch {
            let best = if target_q_values[row * 2 + 1] > target_q_values[row * 2] {
                target_q_values[row * 2 + 1]
            } else {
                target_q_values[row * 2]
            };
            let label = rewards[row] + if terminals[row] { 0.0 } else { discount * best };
            let idx = row * 2 + selected_actions[row];
            grad_q[idx] = (2.0 / (batch as f32)) * (q_values[idx] - label);
        }

        deep_goose::dense_backward_f32_into(
            hidden1, w2, grad_q, grad_hidden, grad_w2, grad_b2, batch, 16, 2
        );
        deep_goose::relu_backward_f32_into(pre1, grad_hidden, grad_pre1);
        deep_goose::dense_backward_f32_into(
            obs_batch, w1, grad_pre1, grad_obs, grad_w1, grad_b1, batch, 8, 16
        );

        deep_goose::sgd_step_f32(w1, grad_w1, learning_rate);
        deep_goose::sgd_step_f32(b1, grad_b1, learning_rate);
        deep_goose::sgd_step_f32(w2, grad_w2, learning_rate);
        deep_goose::sgd_step_f32(b2, grad_b2, learning_rate);
        learner_updates += 1;

        if learner_updates % 100 == 0 {
            for i in 0..128 { target_w1[i] = w1[i]; }
            for i in 0..16 { target_b1[i] = b1[i]; }
            for i in 0..32 { target_w2[i] = w2[i]; }
            for i in 0..2 { target_b2[i] = b2[i]; }
        }
    }
    let elapsed_ns = clock_ns() - started;

    // Independent greedy evaluation; not timed. Initial state is 0.
    var evaluate_state: i64 = 0;
    var evaluate_steps: i64 = 0;
    var initial_right_q: f32 = 0.0;
    for step in 0..64 {
        deep_goose::fill_f32(online_obs, 0.0);
        online_obs[evaluate_state] = 1.0;
        deep_goose::dense_forward_f32_into(online_obs,w1,b1,online_pre,1,8,16);
        deep_goose::relu_f32_into(online_pre,online_hidden);
        deep_goose::dense_forward_f32_into(online_hidden,w2,b2,online_q,1,16,2);
        if step == 0 { initial_right_q = online_q[1]; }
        let action = if online_q[1] >= online_q[0] { 1 } else { 0 };
        evaluate_state = if action == 0 {
            if evaluate_state > 0 { evaluate_state - 1 } else { 0 }
        } else {
            if evaluate_state < 7 { evaluate_state + 1 } else { 7 }
        };
        evaluate_steps += 1;
        if evaluate_state == 7 { break; }
    }

    assert(completed > 100);
    assert(evaluate_state == 7 && evaluate_steps == 7);
    assert(initial_right_q > 0.1);
    print("DQN_RESULT,goose,", training_steps, ",", elapsed_ns, ",",
          learner_updates, ",", completed, ",", evaluate_steps, ",", initial_right_q);
}
