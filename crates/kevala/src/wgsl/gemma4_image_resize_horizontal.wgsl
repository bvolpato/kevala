struct Params {
  input_width: u32,
  input_height: u32,
  output_width: u32,
  output_height: u32,
  output_count: u32,
  patch_columns: u32,
  horizontal_byte_count: u32,
  reserved: u32,
};

const Q22_HALF: i32 = 2097152;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> source_rgba: array<u32>;
@group(0) @binding(2) var<storage, read> x_meta: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> x_weights: array<i32>;
@group(0) @binding(4) var<storage, read_write> horizontal_rgb: array<u32>;

fn source_channel(y: u32, x: u32, channel: u32) -> i32 {
  let rgba = source_rgba[y * p.input_width + x];
  return i32((rgba >> (channel * 8u)) & 255u);
}

fn horizontal_byte(y: u32, x: u32, channel: u32) -> u32 {
  let item = x_meta[x];
  var sum = Q22_HALF;
  for (var tap = 0u; tap < item.y; tap += 1u) {
    sum += source_channel(y, item.x + tap, channel) * x_weights[item.z + tap];
  }
  return u32(clamp(sum >> 22, 0, 255));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word = gid.x;
  let word_count = (p.horizontal_byte_count + 3u) / 4u;
  if (word >= word_count) { return; }
  var packed = 0u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    let scalar = word * 4u + lane;
    if (scalar < p.horizontal_byte_count) {
      let row = scalar / (p.output_width * 3u);
      let in_row = scalar % (p.output_width * 3u);
      let x = in_row / 3u;
      let channel = in_row % 3u;
      packed |= horizontal_byte(row, x, channel) << (lane * 8u);
    }
  }
  horizontal_rgb[word] = packed;
}
