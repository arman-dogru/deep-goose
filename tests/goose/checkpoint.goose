// Integration test: save/load, shape mismatch, missing and malformed files.
import deep_goose from "../../src";
import os;

fn main() {
    let path = "build/deep_goose_checkpoint.bin";
    let corrupt_path = "build/deep_goose_checkpoint_corrupt.bin";
    var weights: f32[3] = [1.0, -2.0, 3.5];
    var bias: f32[1] = [0.25];
    assert(deep_goose::save_dense_parameters_f32(path, weights, bias));
    weights[0] = 77.0;
    weights[1] = 88.0;
    weights[2] = 99.0;
    bias[0] = -88.0;
    assert(deep_goose::load_dense_parameters_f32(path, weights, bias));
    assert(weights[0] == 1.0 && weights[1] == -2.0 && weights[2] == 3.5);
    assert(bias[0] == 0.25);

    // Exact count mismatch must not mutate caller's destination.
    var wrong_weights: f32[2] = [7.0, 8.0];
    assert(!deep_goose::load_dense_parameters_f32(path, wrong_weights, bias));
    assert(wrong_weights[0] == 7.0 && wrong_weights[1] == 8.0);

    let broken: u8[3] = [68, 71, 65];
    assert(write_file(corrupt_path, broken));
    assert(!deep_goose::load_dense_parameters_f32(corrupt_path, weights, bias));
    assert(!deep_goose::load_dense_parameters_f32("build/nonexistent_404.bin", weights, bias));
    assert(weights[0] == 1.0 && weights[1] == -2.0 && bias[0] == 0.25);

    assert(delete_file(path));
    assert(delete_file(corrupt_path));
    print("PASS versioned dense-parameter checkpoint roundtrip and malformed-file rejection");
}
