// Kev trunk: the per-pass globals and the packed segment table.
struct Globals { T: u32, R: u32, S: u32, stage: u32 }
// a packed segment: tokens start..start+len continue carry slot parent (keys at KV rows
// pstart..pstart+plen) and, in stage 1, leave their carry in slot dst, own keys from KV row kvdst
struct Seg { start: u32, len: u32, parent: u32, pstart: u32, plen: u32, dst: u32, kvdst: u32, _b: u32 }
@group(0) @binding(0) var<uniform> g: Globals;
