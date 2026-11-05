// Portable versioned f32 parameter checkpoint for one dense layer.
// File format: "DGAI", u32 version=1, u32 weight count, u32 bias count,
// then little-endian f32 weights and biases. No optimizer state.
// Reader rejects malformed magic, version, sizes, and truncated files.
// This format does not contain a cryptographic integrity check.
import os;
import binary;
namespace deep_goose;

fn save_dense_parameters_f32(path: u8[:], weights: const f32[:],
                             bias: const f32[:]) -> bool {
    if weights.len > 1048576 || bias.len > 1048576 { return false; }
    var bytes: u8[>..] = [];
    bytes.push(68);
    bytes.push(71);
    bytes.push(65);
    bytes.push(73);
    binary::write_u32le(bytes, 1);
    binary::write_u32le(bytes, weights.len as u32);
    binary::write_u32le(bytes, bias.len as u32);
    for value in weights { binary::write_f32le(bytes, value); }
    for value in bias { binary::write_f32le(bytes, value); }
    write_file(path, bytes)
}

fn load_dense_parameters_f32(path: u8[:], weights: f32[:], bias: f32[:]) -> bool {
    if weights.len > 1048576 || bias.len > 1048576 { return false; }
    var bytes: u8[>..] = [];
    if !read_file(path, bytes) { return false; }
    if bytes.len != 16 + 4 * (weights.len + bias.len) { return false; }
    if bytes[0] != 68 || bytes[1] != 71 || bytes[2] != 65 || bytes[3] != 73 {
        return false;
    }
    let version, ok_version = binary::read_u32le(bytes, 4);
    let count_weights, ok_weights = binary::read_u32le(bytes, 8);
    let count_bias, ok_bias = binary::read_u32le(bytes, 12);
    if !ok_version || !ok_weights || !ok_bias { return false; }
    if version != 1 || count_weights != weights.len as u32 ||
       count_bias != bias.len as u32 { return false; }

    // Once exact length and counts are checked all offsets are valid.
    for i in 0..weights.len {
        let value, ok = binary::read_f32le(bytes, 16 + i * 4);
        if !ok { return false; }
        weights[i] = value;
    }
    for i in 0..bias.len {
        let value, ok = binary::read_f32le(bytes, 16 + (weights.len + i) * 4);
        if !ok { return false; }
        bias[i] = value;
    }
    return true;
}
