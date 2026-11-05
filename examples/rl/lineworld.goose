// Native Goose tabular Q-learning in a deterministic 8-state LineWorld.
// Episodes start in state 0; actions 0/1 move left/right. State 7 is terminal.
// Output is timed on-policy interaction + updates; final greedy evaluation is excluded.
// The LCG and exploration policy match benchmarks/c/lineworld.c exactly.
import deep_goose from "../../src";
import os;

fn main() {
    let episodes: i64 = 3000;
    let max_steps: i64 = 64;
    var q: f32[16] = [0.0; 16];
    var seed: i64 = 12345;
    var transitions: i64 = 0;
    var completed: i64 = 0;

    let started = clock_ns();
    for episode in 0..episodes {
        var state: i64 = 0;
        for step in 0..max_steps {
            seed = (seed * 1664525 + 1013904223) % 4294967296;
            let explore = (seed % 1000) < 100;
            let action: i64 = if explore {
                if (seed % 2) == 0 { 0 } else { 1 }
            } else {
                if q[state * 2 + 1] >= q[state * 2] { 1 } else { 0 }
            };

            let next_state = if action == 0 {
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
            deep_goose::q_learning_update_f32(
                q, 8, 2, transition, 0.2, 0.95
            );
            transitions += 1;
            if terminal {
                completed += 1;
                break;
            }
            state = next_state;
        }
    }
    let elapsed_ns = clock_ns() - started;

    // Learned greedy policy should reach the goal in the shortest path.
    var state: i64 = 0;
    var evaluation_steps: i64 = 0;
    for step in 0..max_steps {
        let action = if q[state * 2 + 1] >= q[state * 2] { 1 } else { 0 };
        state = if action == 0 {
            if state > 0 { state - 1 } else { 0 }
        } else {
            if state < 7 { state + 1 } else { 7 }
        };
        evaluation_steps += 1;
        if state == 7 { break; }
    }
    assert(state == 7 && evaluation_steps == 7 && completed == episodes);
    print("RL_RESULT,goose,", episodes, ",", elapsed_ns, ",",
          transitions, ",", q[1], ",", q[13], ",", evaluation_steps);
}
