// Versioned f32 dense-parameter checkpoint for Goose v2026.3.
// Format: magic "DGAI", one-byte version (1), two u32 little-endian counts,
// then ONE Goose-framed f32 array containing weights followed by biases.
// This is a Goose data format, NOT a cross-framework interchange standard.
// Reader validates magic, version, shape and the complete array frame.
// Note: the format does not authenticate or checksum numeric values.
import os;
import binary;
namespace deep_goose;

fn write_u32_bytes(out: u8[>..]&, count: i64) {
    assert(count >= 0 && count <= 1048576);
    out.push((count % 256) as u8);
    out.push(((count / 256) % 256) as u8);
    out.push(((count / 65536) % 256) as u8);
    out.push(((count / 16777216) % 256) as u8);
}

fn save_dense_parameters_f32(path: u8[:], weights: const f32[:],
                             bias: const f32[:]) -> bool {
    if weights.len > 1048576 || bias.len > 1048576 { return false; }
    var combined: f32[>..] = [];
    for value in weights { combined.push(value); }
    for value in bias { combined.push(value); }
    var output: u8[>..] = [];
    output.push(68);
    output.push(71);
    output.push(65);
    output.push(73);
    output.push(1);
    write_u32_bytes(output, weights.len);
    write_u32_bytes(output, bias.len);
    to_bytes(combined, output);
    write_file(path, output)
}

fn load_dense_parameters_f32(path: u8[:], weights: f32[:], bias: f32[:]) -> bool {
    if weights.len > 1048576 || bias.len > 1048576 { return false; }
    var input: u8[>..] = [];
    if !read_file(path, input) { return false; }
    if input.len < 14 { return false; }
    if input[0] != 68 || input[1] != 71 || input[2] != 65 ||
       input[3] != 73 || input[4] != 1 { return false; }
    let count_weights, ok_weights = binary::read_u32le(input, 5);
    let count_bias, ok_bias = binary::read_u32le(input, 9);
    if !ok_weights || !ok_bias { return false; }
    if count_weights != weights.len as u32 || count_bias != bias.len as u32 {
        return false;
    }
    var combined, ok = from_bytes<f32[>..]>(input[13..]);
    if !ok || combined.len != weights.len + bias.len { return false; }
    for i in 0..weights.len { weights[i] = combined[i]; }
    for i in 0..bias.len { bias[i] = combined[weights.len + i]; }
    return true;
}
