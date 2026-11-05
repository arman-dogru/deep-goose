// Dependency-free tabular RL primitives, independent of neural-network code.
namespace deep_goose;

struct Transition {
    state: i64,
    action: i64,
    reward: f32,
    next_state: i64,
    terminated: bool,
    truncated: bool
}

struct ReplayBuffer {
    capacity: i64,
    cursor: i64,
    entries: Transition[>..]
}

fn new_replay(capacity: i64) -> ReplayBuffer {
    assert(capacity > 0 && capacity <= 1048576);
    ReplayBuffer { capacity: capacity, cursor: 0, entries: [] }
}

fn replay_push(buffer: ReplayBuffer&, transition: Transition) {
    if buffer.entries.len < buffer.capacity {
        buffer.entries.push(transition);
    } else {
        buffer.entries[buffer.cursor] = transition;
    }
    buffer.cursor = (buffer.cursor + 1) % buffer.capacity;
}

fn replay_get(buffer: ReplayBuffer&, index: i64) -> Transition {
    assert(index >= 0 && index < buffer.entries.len);
    buffer.entries[index]
}

fn q_learning_update_f32(q: f32[:], states: i64, actions: i64,
                         transition: Transition, alpha: f32, gamma: f32) {
    assert(states > 0 && actions > 0 && states <= 1048576 && actions <= 1048576);
    assert(q.len == states * actions);
    assert(transition.state >= 0 && transition.state < states);
    assert(transition.next_state >= 0 && transition.next_state < states);
    assert(transition.action >= 0 && transition.action < actions);
    assert(alpha > 0.0 && alpha <= 1.0 && gamma >= 0.0 && gamma <= 1.0);
    let idx = transition.state * actions + transition.action;
    var target = transition.reward;
    // Bootstrap through time-limit truncation, but not true termination.
    if !transition.terminated {
        let start = transition.next_state * actions;
        var best = q[start];
        for a in 1..actions {
            if q[start + a] > best { best = q[start + a]; }
        }
        target += gamma * best;
    }
    q[idx] += alpha * (target - q[idx]);
}
