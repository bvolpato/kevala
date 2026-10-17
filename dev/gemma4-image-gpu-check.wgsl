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
fn resize_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
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

@group(0) @binding(5) var<storage, read> y_meta: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> y_weights: array<i32>;
@group(0) @binding(7) var<storage, read> vertical_input: array<u32>;
@group(0) @binding(8) var<storage, read> normalized_lut: array<f32>;
@group(0) @binding(9) var<storage, read_write> pixels: array<f32>;

fn horizontal_value(y: u32, x: u32, channel: u32) -> i32 {
  let scalar = (y * p.output_width + x) * 3u + channel;
  let packed = vertical_input[scalar / 4u];
  return i32((packed >> ((scalar % 4u) * 8u)) & 255u);
}

@compute @workgroup_size(128)
fn resize_vertical_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= p.output_count) { return; }
  let patch_index = index / 768u;
  let within_patch = index % 768u;
  let channel = within_patch % 3u;
  let pixel = within_patch / 3u;
  let x = (patch_index % p.patch_columns) * 16u + pixel % 16u;
  let y = (patch_index / p.patch_columns) * 16u + pixel / 16u;
  let item = y_meta[y];
  var sum = Q22_HALF;
  for (var tap = 0u; tap < item.y; tap += 1u) {
    sum += horizontal_value(item.x + tap, x, channel) * y_weights[item.z + tap];
  }
  let byte = u32(clamp(sum >> 22, 0, 255));
  pixels[index] = normalized_lut[byte];
}
