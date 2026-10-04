// The `.kevala` pack as the JavaScript side reads it: the header, the size checks that run before
// any byte crosses into WebAssembly, and the coordinator/GPU split of whole tensors.
//
// A GPU trunk can exceed wasm32's address space, so source ranges stay JavaScript numbers here
// and only the rebased coordinator pack is passed to WebAssembly.

const ALIGN = 64;
const MAGIC = "KVLA";
const FORMAT_VERSION = 1;
const dec = new TextDecoder();
// Rust's aligned allocations are limited by isize::MAX, including alignment padding.
const MAX_WASM_PACK_BYTES = 0x7fffffc0;
const enc = new TextEncoder();

function integer(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`pack: ${label} must be a nonnegative safe integer`);
  return value;
}

const add = (a, b, label) => integer(a + b, label);
const align = (n) => integer(Math.ceil(n / ALIGN) * ALIGN, "aligned offset");
const trunk = (name) => name.startsWith("L.") || name.startsWith("enc.") || name.startsWith("head.");

/** Bytes of the 16-byte prefix plus the JSON header it announces. `bytes` holds at least the prefix. */
export function packHeaderLength(bytes) {
  return 16 + new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(8, true);
}

/** The JSON header of a pack or sub-pack whose leading bytes are `bytes`. */
export function parsePackHeader(bytes) {
  if (bytes.byteLength < 16 || dec.decode(bytes.subarray(0, 4)) !== MAGIC) throw new Error("not a .kevala pack");
  const version = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, true);
  if (version !== FORMAT_VERSION) throw new Error(`unsupported .kevala version ${version}, this build reads ${FORMAT_VERSION}`);
  const end = packHeaderLength(bytes);
  if (bytes.byteLength < end) throw new Error("pack ended before its header");
  return JSON.parse(dec.decode(bytes.subarray(16, end)));
}

/**
 * Reads the header from the front of a pack arriving as chunks, keeping every byte it consumed:
 * `head` (all of them, which may run past the header), the parsed `header`, and `headerBytes`.
 */
export async function readPackHead(chunks) {
  let head = new Uint8Array(0);
  const need = () => (head.byteLength < 16 ? 16 : packHeaderLength(head));
  while (head.byteLength < need()) {
    const { done, value } = await chunks.next();
    if (done) throw new Error("pack ended before its header");
    const grown = new Uint8Array(head.byteLength + value.byteLength);
    grown.set(head);
    grown.set(value, head.byteLength);
    head = grown;
  }
  return { head, header: parsePackHeader(head), headerBytes: head.subarray(0, packHeaderLength(head)) };
}

/** Validates byte ranges before any of them cross a wasm32 boundary. */
export function packSize(header, headerBytes) {
  const start = align(integer(headerBytes, "header size"));
  const ranges = [];
  const range = (offset, size, label) => {
    integer(offset, `${label} offset`);
    integer(size, `${label} size`);
    const end = add(offset, size, `${label} end`);
    if (size) {
      if (offset < start || offset % ALIGN) throw new Error(`pack: ${label} has an unaligned or overlapping header offset`);
      ranges.push({ start: offset, end, label });
    }
    return end;
  };
  let total = Math.max(start, range(header.tokenizer?.offset, header.tokenizer?.size, "tokenizer"));
  if (!Array.isArray(header.tensors)) throw new Error("pack: missing tensors");
  const names = new Set();
  for (const tensor of header.tensors) {
    const { name, dtype, shape } = tensor;
    if (typeof name !== "string" || !name || names.has(name)) throw new Error("pack: tensor names must be nonempty and unique");
    names.add(name);
    if (!Array.isArray(shape)) throw new Error(`pack: ${name} has no shape`);
    const elements = shape.reduce((n, dim) => integer(n * integer(dim, `${name} dimension`), `${name} element count`), 1);
    if (dtype === "f32") {
      if (tensor.size !== integer(elements * 4, `${name} byte count`)) throw new Error(`pack: ${name} has an inconsistent size`);
    } else if (dtype === "q8") {
      const block = integer(tensor.block, `${name} block`);
      if (shape.length !== 2 || !block || shape[1] % block || tensor.size !== elements ||
          tensor.scales_size !== integer((elements / block) * 4, `${name} scales byte count`)) {
        throw new Error(`pack: ${name} has an inconsistent q8 layout`);
      }
      total = Math.max(total, range(tensor.scales_offset, tensor.scales_size, `${name} scales`));
    } else throw new Error(`pack: ${name} has unsupported dtype ${dtype}`);
    total = Math.max(total, range(tensor.offset, tensor.size, name));
  }
  ranges.sort((a, b) => a.start - b.start);
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i].start < ranges[i - 1].end) throw new Error(`pack: ${ranges[i].label} overlaps ${ranges[i - 1].label}`);
  }
  return total;
}

export function assertWasmPackSize(bytes, role = "model") {
  integer(bytes, `${role} size`);
  if (bytes > MAX_WASM_PACK_BYTES) {
    const advice = role === "coordinator" ? "Use the native Kevala CLI for this model." : "Use WebGPU or the native Kevala CLI for this model.";
    throw Object.assign(new Error(`The ${role} pack needs ${(bytes / 2 ** 30).toFixed(2)} GiB, exceeding the WebAssembly allocation limit (under 2 GiB). ${advice}`), { code: "WASM_PACK_TOO_LARGE" });
  }
}

function subset(header, keepTrunk, isTrunk) {
  const tensors = header.tensors.filter((t) => isTrunk(t.name) === keepTrunk).map((t) => ({
    name: t.name, dtype: t.dtype, shape: [...t.shape], offset: t.offset, size: t.size,
    ...(t.dtype === "q8" ? { block: t.block, scales_offset: t.scales_offset, scales_size: t.scales_size } : {}),
  }));
  const tokenizer = { offset: 0, size: keepTrunk ? 0 : header.tokenizer.size };
  const parts = [];
  if (!keepTrunk) parts.push({ src: header.tokenizer.offset, size: tokenizer.size, info: tokenizer, field: "offset" });
  for (const info of tensors) {
    parts.push({ src: info.offset, size: info.size, info, field: "offset" });
    if (info.dtype === "q8") parts.push({ src: info.scales_offset, size: info.scales_size, info, field: "scales_offset" });
  }
  // GpuWeights consumes destination ranges in order, even when source tensor metadata is unordered.
  parts.sort((a, b) => a.src - b.src);
  let room = 0;
  for (;;) {
    let at = align(16 + room);
    tokenizer.offset = at;
    const pieces = [];
    for (const part of parts) {
      part.info[part.field] = at;
      pieces.push(part.src, at, part.size, 1, 0);
      at = align(add(at, part.size, "subset size"));
    }
    const json = enc.encode(JSON.stringify({ format: "kevala", version: FORMAT_VERSION, model: header.model, config: header.config, tokenizer, tensors }));
    if (json.byteLength > room) {
      room = json.byteLength + 256;
      continue;
    }
    const prefix = new Uint8Array(16 + room);
    prefix.set(enc.encode(MAGIC));
    const view = new DataView(prefix.buffer);
    view.setUint32(4, FORMAT_VERSION, true);
    view.setUint32(8, room, true);
    prefix.fill(32, 16);
    prefix.set(json, 16);
    return { prefix, total: at, pieces };
  }
}

/**
 * The coordinator and GPU layouts of a pack: whole tensors, placed by `isTrunk(name)`. The default
 * rule is the one of Rust's `pack::coord_layout` and `pack::trunk_layout`. Unlike those, the
 * source offsets here are not limited to u32.
 */
export function gpuLayouts(header, headerBytes, isTrunk = trunk) {
  packSize(header, headerBytes);
  const coord = subset(header, false, isTrunk);
  assertWasmPackSize(coord.total, "coordinator");
  return [coord, subset(header, true, isTrunk)];
}
