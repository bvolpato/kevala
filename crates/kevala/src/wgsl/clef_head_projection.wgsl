// Clef tile-0 F32 projection. The first staged request validates every output bit.
struct Params { T: u32, N: u32, K: u32, output_offset: u32, tail_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> ODD_RECORD_TAIL: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

fn mac(acc: vec4<f32>, x: vec4<f32>, w: vec4<f32>) -> vec4<f32> {
  return fma(x, w, acc);
}
fn hsum(acc: vec4<f32>) -> f32 {
  return (acc.x + acc.z) + (acc.y + acc.w);
}

var<workgroup> xs: array<vec4<f32>, 144>;
var<workgroup> ws: array<vec4<f32>, 128>;

fn store_bulk(acc: vec4<f32>, row: u32, col: u32) {
  if (row < p.T && col < p.N) {
    if (ODD_RECORD_TAIL[row] == 0u) {
      Y[p.output_offset + row * p.N + col] = hsum(acc) + B[col];
    }
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = gid.y * 16u;
  let n0 = gid.x * 16u;
  let lr = li / 4u;
  let lq = li % 4u;
  let row = m0 + lr;
  let col = n0 + lr;
  let load_sw = lr & 7u;
  let sw = lid.x & 7u;
  let stride = p.K / 4u;
  var c00 = vec4<f32>(0.0);
  var c01 = vec4<f32>(0.0);
  var c10 = vec4<f32>(0.0);
  var c11 = vec4<f32>(0.0);
  for (var block = 0u; block < p.K / 32u; block++) {
    var x0 = vec4<f32>(0.0);
    var x1 = vec4<f32>(0.0);
    var w0 = vec4<f32>(0.0);
    var w1 = vec4<f32>(0.0);
    if (row < p.T) {
      let base = row * stride + block * 8u + lq * 2u;
      x0 = X[base]; x1 = X[base + 1u];
    }
    if (col < p.N) {
      let base = col * stride + block * 8u + lq * 2u;
      w0 = W[base]; w1 = W[base + 1u];
    }
    xs[lr * 9u + lq * 2u] = x0;
    xs[lr * 9u + lq * 2u + 1u] = x1;
    ws[lr * 8u + ((lq * 2u) ^ load_sw)] = w0;
    ws[lr * 8u + ((lq * 2u + 1u) ^ load_sw)] = w1;
    workgroupBarrier();
    {
      let kq = 0u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 1u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 2u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 3u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 4u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 5u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 6u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    {
      let kq = 7u;
      let a0 = xs[lid.y * 9u + kq];
      let a1 = xs[(lid.y + 8u) * 9u + kq];
      let b0 = ws[lid.x * 8u + (kq ^ sw)];
      let b1 = ws[(lid.x + 8u) * 8u + (kq ^ sw)];
      c00 = mac(c00, a0, b0);
      c01 = mac(c01, a0, b1);
      c10 = mac(c10, a1, b0);
      c11 = mac(c11, a1, b1);
    }
    workgroupBarrier();
  }
  store_bulk(c00, m0 + lid.y, n0 + lid.x);
  store_bulk(c01, m0 + lid.y, n0 + lid.x + 8u);
  store_bulk(c10, m0 + lid.y + 8u, n0 + lid.x);
  store_bulk(c11, m0 + lid.y + 8u, n0 + lid.x + 8u);
}
