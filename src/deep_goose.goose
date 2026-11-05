// Public package entrypoint for Deep Goose.
// Import in a root program with: import deep_goose from "../../src";
import ops.cpu;
import ops.classification;
import nn.checkpoint;
import rl.tabular;
namespace deep_goose;

fn version() -> const u8[:] { "0.1.0-dev" }
