struct P { rows: u32, heads: u32, dim: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> C: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let at = g.x;
  if (at >= p.rows * p.heads * p.dim) { return; }
  let row = at / (p.heads * p.dim);
  let col = at % p.dim;
  let spatial = p.dim / 2u;
  let half = spatial / 2u;
  let axis = col / spatial;
  let part = col % spatial;
  let pair = part % half;
  let cs = C[row * spatial + axis * half + pair];
  let partner = select(at + half, at - half, part >= half);
  let rotated = select(-X[partner], X[partner], part >= half);
  Y[at] = X[at] * cs.x + rotated * cs.y;
}
