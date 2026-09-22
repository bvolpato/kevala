// Laya trunk: the per-pass globals every kernel reads.
struct Globals { T: u32, R: u32, S: u32, _b: u32 }
@group(0) @binding(0) var<uniform> g: Globals;
