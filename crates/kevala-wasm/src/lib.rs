//! The C ABI the JavaScript runtime calls. One instance plays one role:
//!
//! - engine: a whole pack, answers requests on its own (`kevala_engine_load` + `kevala_decide`);
//! - coordinator: the coordinator sub-pack, tokenizes and scores while shards or the GPU run the
//!   transformer layers (`kevala_prepare`, `kevala_embed`, `kevala_bridge`, `kevala_finish`);
//! - shard: one tensor-parallel slice of the layers (`kevala_shard_load`, `kevala_shard_step`).
//!
//! Strings and results come back through `kevala_out_ptr/len`, errors through `kevala_error_ptr/len`.
//! Every call that can fail returns 0 on success and 1 on error.
//!
//! # Safety
//!
//! Every export that takes a pointer is `unsafe`. Unless its own documentation says otherwise, the
//! pointer must be non-null, aligned for its element type, and valid for reads of the given number
//! of elements until the call returns. Exports that take ownership say which allocator the buffer
//! must come from. A WebAssembly instance runs on one thread, so the exports are not reentrant.

use kevala::content::Request;
use kevala::gemma4::Gemma4Engine;
use kevala::json::Value;
use kevala::kev::KevEngine;
use kevala::laya::model::{self, Batch, Scratch, Seg, ShardPlan, Trunk};
use kevala::laya::{Engine, Prepared};
use kevala::pack::{coord_layout, trunk_layout};
use kevala::runtime::{self, Model};
use kevala::simd::F4;
use kevala::store::{load_store, AlignedBuf};

#[cfg(feature = "cpu-bench")]
mod cpu_bench;

mod cpu_tune;

#[derive(Default)]
struct State {
    model: Option<Box<dyn Model>>,
    shard: Option<Trunk>,
    prepared: Option<(Prepared, Vec<Value>)>,
    batch: Batch,
    x: Vec<f32>,
    partial: Vec<f32>,
    reduce_scratch: Vec<f32>,
    scratch: Scratch,
    out: Vec<u8>,
    err: String,
    plan: Option<kevala::laya::convert::Plan>,
    pack: Option<AlignedBuf>,
    kev_batch: Option<(Vec<kevala::kev::Encoded>, Vec<Vec<kevala::kev::KevQuestion>>)>,
    kev_convert: Option<(kevala::kev::convert::KevConvert, Vec<String>)>,
    kev_ids: [Vec<u32>; 2],
    gemma4_prepared: Option<kevala::gemma4::Prepared>,
}

static mut STATE: Option<State> = None;

#[allow(static_mut_refs)]
fn st() -> &'static mut State {
    // SAFETY: a WebAssembly instance runs on one thread
    unsafe { STATE.get_or_insert_with(State::default) }
}

/// The loaded model as a concrete family, for the backend-specific entry points.
fn family<T: 'static>(what: &str) -> Result<&'static mut T, String> {
    st().model
        .as_mut()
        .ok_or("no model loaded")?
        .as_any()
        .downcast_mut::<T>()
        .ok_or_else(|| format!("the loaded model is not {what}"))
}

fn done(r: Result<(), String>) -> u32 {
    match r {
        Ok(()) => 0,
        Err(e) => {
            st().err = e;
            1
        }
    }
}

/// # Safety
/// `ptr` must be valid for reads of `len` elements for as long as the returned slice is used.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    std::slice::from_raw_parts(ptr, len)
}

/// # Safety
/// As for [`slice`].
unsafe fn input<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    std::str::from_utf8(slice(ptr, len)).map_err(|_| "input is not UTF-8".to_string())
}

/// # Safety
/// As for [`slice`].
unsafe fn json_input(ptr: *const u8, len: usize) -> Result<Value, String> {
    Value::parse(input(ptr, len)?).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn kevala_init() {
    std::panic::set_hook(Box::new(|info| {
        st().err = format!("kevala panicked: {info}");
    }));
}

#[no_mangle]
pub extern "C" fn kevala_alloc(len: usize) -> *mut u8 {
    AlignedBuf::new(len).into_raw().0
}

/// # Safety
/// `ptr`/`len` must come from `kevala_alloc` and not have been handed to a loader.
#[no_mangle]
pub unsafe extern "C" fn kevala_free(ptr: *mut u8, len: usize) {
    drop(AlignedBuf::from_raw(ptr, len));
}

#[no_mangle]
pub extern "C" fn kevala_out_ptr() -> *const u8 {
    st().out.as_ptr()
}

#[no_mangle]
pub extern "C" fn kevala_out_len() -> usize {
    st().out.len()
}

#[no_mangle]
pub extern "C" fn kevala_error_ptr() -> *const u8 {
    st().err.as_ptr()
}

#[no_mangle]
pub extern "C" fn kevala_error_len() -> usize {
    st().err.len()
}

fn put_u32(out: &mut Vec<u8>, v: usize) {
    out.extend_from_slice(&(v as u32).to_le_bytes());
}

/// Adds shard partials in their original order, preserving f32 rounding.
#[inline]
fn add_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0;
    unsafe {
        while i + 4 <= n {
            (F4::load(dst.as_ptr().add(i)) + F4::load(src.as_ptr().add(i))).store(dst.as_mut_ptr().add(i));
            i += 4;
        }
    }
    while i < n {
        dst[i] += src[i];
        i += 1;
    }
}

fn write_layout(out: &mut Vec<u8>, l: &kevala::pack::Layout) {
    put_u32(out, l.prefix.len());
    out.extend_from_slice(&l.prefix);
    put_u32(out, l.total);
    put_u32(out, l.pieces.len());
    for p in &l.pieces {
        for v in [p.src, p.dst, p.len, p.rows, p.stride] {
            put_u32(out, v);
        }
    }
}

/// Given the pack header bytes, writes the coordinator layout followed by `count` shard layouts:
/// per layout `u32 prefix_len, prefix, u32 total, u32 n, n * (src, dst, len, rows, stride)`.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_layouts(ptr: *const u8, len: usize, count: usize) -> u32 {
    done((|| {
        let h = kevala::pack::parse_header(slice(ptr, len))?;
        let mut out = Vec::new();
        write_layout(&mut out, &coord_layout(&h)?);
        if h.config().get("arch").and_then(Value::as_str) == Some("kev") {
            // no tensor-parallel shards for Kev: the whole trunk, for the GPU
            write_layout(&mut out, &trunk_layout(&h)?);
            st().out = out;
            return Ok(());
        }
        let cfg = model::Config::from_json(h.config())?;
        for i in 0..count {
            write_layout(&mut out, &model::shard_layout(&cfg, &h, ShardPlan { index: i, count })?);
        }
        st().out = out;
        Ok(())
    })())
}

/// Takes ownership of a buffer from `kevala_alloc` holding a whole pack of any family this build
/// knows, or a coordinator sub-pack. Writes `{"arch", "model", "modalities"}`.
///
/// # Safety
/// `ptr`/`len` must come from `kevala_alloc`. The caller must not use or free the buffer afterwards.
#[no_mangle]
pub unsafe extern "C" fn kevala_engine_load(ptr: *mut u8, len: usize) -> u32 {
    done((|| {
        let m = runtime::load(AlignedBuf::from_raw(ptr, len))?;
        let s = st();
        s.out = Value::Object(vec![
            ("arch".into(), Value::Str(m.arch().into())),
            ("model".into(), m.info().clone()),
            ("modalities".into(), Value::Array(m.modalities().iter().map(|x| Value::Str(x.name().into())).collect())),
        ])
        .to_json()
        .into_bytes();
        s.model = Some(m);
        Ok(())
    })())
}

/// # Safety
/// As for [`slice`].
unsafe fn prepare_json(ptr: *const u8, len: usize) -> Result<(), String> {
    let e: &Engine = family("a laya model")?;
    st().prepared = Some(e.prepare_requests(&Request::parse_many(&json_input(ptr, len)?)?)?);
    Ok(())
}

fn respond() -> Result<(), String> {
    let s = st();
    let e: &Engine = family("a laya model")?;
    let (p, qs) = s.prepared.take().ok_or("nothing prepared")?;
    let outs = e.coord.score(&s.x, &p.batch);
    let scored = e.score(&p, &outs);
    let qrefs: Vec<&Value> = qs.iter().collect();
    s.out = Value::Array(e.respond(&p, &scored, &qrefs)).to_json().into_bytes();
    Ok(())
}

/// Answers `{"state", "parts"?, "questions"}` or `{"requests": [...]}` with whichever model is
/// loaded, in one forward pass. Writes a JSON array with one response per request.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_decide(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let rs = Request::parse_many(&json_input(ptr, len)?)?;
        let s = st();
        let out = s.model.as_mut().ok_or("no model loaded")?.decide(&rs)?;
        s.out = Value::Array(out).to_json().into_bytes();
        Ok(())
    })())
}

/// Tokenizes a request for an external trunk. Writes the segment table as u32s:
/// `tokens, segments, then per segment start, len, qtype, markers, marker positions...`.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_prepare(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        prepare_json(ptr, len)?;
        let s = st();
        let (p, _) = s.prepared.as_ref().unwrap();
        let mut out = Vec::new();
        put_u32(&mut out, p.batch.tokens());
        put_u32(&mut out, p.batch.segs.len());
        for g in &p.batch.segs {
            for v in [g.start, g.len, g.qtype, g.markers.len()] {
                put_u32(&mut out, v);
            }
            for &m in &g.markers {
                put_u32(&mut out, m);
            }
        }
        s.out = out;
        Ok(())
    })())
}

/// Embeds the prepared batch into the residual buffer and returns it (`tokens * hidden` f32).
#[no_mangle]
pub extern "C" fn kevala_embed() -> *mut f32 {
    let s = st();
    match (family::<Engine>("a laya model").ok(), s.prepared.as_ref()) {
        (Some(e), Some((p, _))) => {
            s.x = e.coord.embed(&p.batch);
            s.x.as_mut_ptr()
        }
        _ => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn kevala_x_ptr() -> *mut f32 {
    st().x.as_mut_ptr()
}

/// Final encoder norm plus the question-type embedding, in place on the residual buffer.
#[no_mangle]
pub extern "C" fn kevala_bridge() -> u32 {
    done((|| {
        let e: &Engine = family("a laya model")?;
        let s = st();
        let (p, _) = s.prepared.as_ref().ok_or("nothing prepared")?;
        e.coord.bridge(&mut s.x, &p.batch);
        Ok(())
    })())
}

/// Scores the residual buffer after the last head layer and writes the JSON responses.
#[no_mangle]
pub extern "C" fn kevala_finish() -> u32 {
    done(respond())
}

/// Takes ownership of a shard sub-pack from `kevala_alloc`.
///
/// # Safety
/// `ptr`/`len` must come from `kevala_alloc`. The caller must not use or free the buffer afterwards.
#[no_mangle]
pub unsafe extern "C" fn kevala_shard_load(ptr: *mut u8, len: usize, primary: u32) -> u32 {
    done((|| {
        let (store, h) = load_store(AlignedBuf::from_raw(ptr, len))?;
        let cfg = model::Config::from_json(h.config())?;
        st().shard = Some(Trunk::new(cfg, std::sync::Arc::new(store), primary != 0)?);
        Ok(())
    })())
}

/// Sets the batch a shard computes on from the table `kevala_prepare` wrote, and sizes the
/// residual buffer. Returns the residual buffer for the caller to fill before each step.
///
/// # Safety
/// `ptr` must be valid for reads of `len` `u32` values that hold a complete `kevala_prepare` table.
#[no_mangle]
pub unsafe extern "C" fn kevala_shard_batch(ptr: *const u32, len: usize) -> *mut f32 {
    let s = st();
    let t = slice(ptr, len);
    let (tokens, n) = (t[0] as usize, t[1] as usize);
    let mut segs = Vec::with_capacity(n);
    let mut i = 2;
    for _ in 0..n {
        let (start, len, qtype, k) = (t[i] as usize, t[i + 1] as usize, t[i + 2] as usize, t[i + 3] as usize);
        segs.push(Seg { start, len, qtype, markers: t[i + 4..i + 4 + k].iter().map(|&m| m as usize).collect() });
        i += 4 + k;
    }
    // shards never look at token ids, only at how many there are
    s.batch = Batch { ids: vec![0; tokens], segs };
    let hidden = s.shard.as_ref().map_or(0, |t| t.cfg.hidden);
    s.x.resize(tokens * hidden, 0.0);
    s.partial.resize(tokens * hidden, 0.0);
    s.x.as_mut_ptr()
}

/// Runs step `step` on the residual buffer and returns this shard's partial update.
#[no_mangle]
pub extern "C" fn kevala_shard_step(step: usize) -> *const f32 {
    let s = st();
    match s.shard.as_ref() {
        Some(t) => {
            t.step(step, &s.x, &s.batch, &mut s.partial, &mut s.scratch);
            s.partial.as_ptr()
        }
        None => std::ptr::null(),
    }
}

/// Reserves the coordinator-side reduction buffers for one tensor-parallel batch.
///
/// The returned buffer is the accumulator. The caller fills it with the local shard partial,
/// then copies each remote partial into `kevala_reduce_partial_ptr` in shard order.
#[no_mangle]
pub extern "C" fn kevala_reduce_prepare(len: usize) -> *mut f32 {
    let s = st();
    s.partial.resize(len, 0.0);
    s.reduce_scratch.resize(len, 0.0);
    s.partial.as_mut_ptr()
}

/// Returns the staging buffer prepared by `kevala_reduce_prepare`.
#[no_mangle]
pub extern "C" fn kevala_reduce_partial_ptr() -> *mut f32 {
    st().reduce_scratch.as_mut_ptr()
}

/// Adds the staged remote shard partial to the accumulator.
#[no_mangle]
pub extern "C" fn kevala_reduce_add_partial() -> u32 {
    done((|| {
        let s = st();
        if s.partial.len() != s.reduce_scratch.len() {
            return Err("reduction buffers are not prepared".to_string());
        }
        add_f32(&mut s.partial, &s.reduce_scratch);
        Ok(())
    })())
}

/// Adds the ordered reduction result to the coordinator residual buffer.
#[no_mangle]
pub extern "C" fn kevala_reduce_finish() -> u32 {
    done((|| {
        let s = st();
        if s.x.len() != s.partial.len() {
            return Err("reduction length does not match residual buffer".to_string());
        }
        add_f32(&mut s.x, &s.partial);
        Ok(())
    })())
}

/// The WGSL source of a GPU kernel specialized as asked: `{ kernel, ...spec }`, with the spec fields
/// that `kevala::gpu::Spec::from_json` reads.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_wgsl(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        st().out = kevala::gpu::wgsl_json(&json_input(ptr, len)?)?.into_bytes();
        Ok(())
    })())
}

/// Starts an in-browser conversion. Inputs are the safetensors file from byte 0 through its JSON
/// header, and the three upstream config files. Allocates the output pack and writes a JSON job
/// list: `{"total", "pack": ptr, "jobs": [[src_offset, src_len], ...]}` in file order.
///
/// # Safety
/// Each pointer must be valid for reads of the length that follows it.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kevala_convert_plan(
    head: *const u8,
    head_len: usize,
    enc: *const u8,
    enc_len: usize,
    agent: *const u8,
    agent_len: usize,
    tok: *const u8,
    tok_len: usize,
    model: *const u8,
    model_len: usize,
    block: usize,
) -> u32 {
    done((|| {
        let head = slice(head, head_len);
        let model = json_input(model, model_len)?;
        let opt = kevala::laya::convert::Options { block, model, keep_f32: Vec::new() };
        let plan = kevala::laya::convert::plan(
            head,
            input(enc, enc_len)?,
            input(agent, agent_len)?,
            input(tok, tok_len)?,
            &opt,
        )?;
        let mut pack = AlignedBuf::new(plan.total);
        pack.as_mut_slice()[..plan.prefix.len()].copy_from_slice(&plan.prefix);
        let jobs = plan
            .jobs
            .iter()
            .map(|j| Value::Array(vec![Value::Int(j.src_offset.to_string()), Value::Int(j.src_len.to_string())]))
            .collect();
        let s = st();
        let ptr = pack.as_mut_ptr() as usize;
        s.out = Value::Object(vec![
            ("total".into(), Value::Int(plan.total.to_string())),
            ("pack".into(), Value::Int(ptr.to_string())),
            ("jobs".into(), Value::Array(jobs)),
        ])
        .to_json()
        .into_bytes();
        s.plan = Some(plan);
        s.pack = Some(pack);
        Ok(())
    })())
}

/// Converts job `i` from its source bytes into the output pack.
///
/// # Safety
/// `src` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_convert_job(i: usize, src: *const u8, len: usize) -> u32 {
    done((|| {
        let s = st();
        let plan = s.plan.as_ref().ok_or("no conversion in progress")?;
        let job = plan.jobs.get(i).ok_or("no such job")?;
        if len != job.src_len {
            return Err(format!("job {i} ({}) needs {} bytes, got {len}", job.src, job.src_len));
        }
        let pack = s.pack.as_mut().ok_or("no conversion in progress")?;
        kevala::laya::convert::run_job(job, slice(src, len), plan.block, pack.as_mut_slice());
        Ok(())
    })())
}

/// Frees the converted pack (after the caller copied it out).
#[no_mangle]
pub extern "C" fn kevala_convert_drop() {
    let s = st();
    s.plan = None;
    s.kev_convert = None;
    s.pack = None;
}

/// Kev on an external trunk: tokenizes a request and writes the batch table as u32s:
/// `R, T1, T2, rows, R x (state start, state len), B, B x (branch start, branch len, request),
/// rows x (readout row in stage 2)`.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_kev_prepare(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let e: &KevEngine = family("a kev model")?;
        let (enc, qs) = e.prepare_requests(&Request::parse_many(&json_input(ptr, len)?)?)?;
        let s = st();
        let mut ids1 = Vec::new();
        let mut ids2 = Vec::new();
        let mut states = Vec::new();
        let mut branches = Vec::new();
        for (ri, en) in enc.iter().enumerate() {
            states.push((ids1.len(), en.state.len()));
            ids1.extend_from_slice(&en.state);
            for b in &en.branches {
                branches.push((ids2.len(), b.ids.len(), ri));
                ids2.extend_from_slice(&b.ids);
            }
        }
        let rows = kevala::kev::KevModel::readout_rows(&enc);
        let mut out = Vec::new();
        for v in [enc.len(), ids1.len(), ids2.len(), rows.len()] {
            put_u32(&mut out, v);
        }
        for (a, b) in &states {
            put_u32(&mut out, *a);
            put_u32(&mut out, *b);
        }
        put_u32(&mut out, branches.len());
        for (a, b, c) in &branches {
            for v in [*a, *b, *c] {
                put_u32(&mut out, v);
            }
        }
        for r in rows {
            put_u32(&mut out, r);
        }
        s.out = out;
        s.kev_ids = [ids1, ids2];
        s.kev_batch = Some((enc, qs));
        Ok(())
    })())
}

/// Token ids of the prepared Kev batch: stage 1 (every state, concatenated) or stage 2.
#[no_mangle]
pub extern "C" fn kevala_kev_ids(stage: usize) -> *const u32 {
    match stage {
        1 | 2 => st().kev_ids[stage - 1].as_ptr(),
        _ => std::ptr::null(),
    }
}

/// Embeds stage 1 (states) or stage 2 (question branches) of the prepared Kev batch into the
/// residual buffer and returns it.
#[no_mangle]
pub extern "C" fn kevala_kev_embed(stage: usize) -> *mut f32 {
    let s = st();
    match (family::<KevEngine>("a kev model").ok(), stage) {
        (Some(e), 1 | 2) => {
            s.x = e.model.embed(&s.kev_ids[stage - 1]);
            s.x.as_mut_ptr()
        }
        _ => std::ptr::null_mut(),
    }
}

/// Reads the pointer rows (`rows * hidden` f32 at `ptr`) and writes the JSON responses.
///
/// # Safety
/// `ptr` must be valid for reads of `n` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn kevala_kev_finish(ptr: *const f32, n: usize) -> u32 {
    done((|| {
        let e: &mut KevEngine = family("a kev model")?;
        let s = st();
        let (enc, qs) = s.kev_batch.take().ok_or("nothing prepared")?;
        let logits = e.model.readout(slice(ptr, n), &enc);
        s.out = Value::Array(e.respond(&enc, &qs, &logits)).to_json().into_bytes();
        Ok(())
    })())
}

/// Writes Gemma's complete prompts as u32s: sequence count, then (length, token IDs) per sequence.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kevala_gemma4_prepare(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        st().gemma4_prepared = None;
        let requests = Request::parse_many(&json_input(ptr, len)?)?;
        let engine: &Gemma4Engine = family("a gemma4 model")?;
        let prepared = engine.prepare_all(&requests)?;
        let mut out = Vec::new();
        put_u32(&mut out, prepared.sequence_count());
        for index in 0..prepared.sequence_count() {
            let ids = prepared.sequence_ids(index);
            put_u32(&mut out, ids.len());
            for &id in ids {
                put_u32(&mut out, id as usize);
            }
        }
        st().out = out;
        st().gemma4_prepared = Some(prepared);
        Ok(())
    })())
}

/// Scores one normalized final hidden row per prepared Gemma prompt.
///
/// # Safety
/// `ptr` must be valid for reads of `n` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn kevala_gemma4_finish(ptr: *const f32, n: usize) -> u32 {
    done((|| {
        let prepared = st().gemma4_prepared.take().ok_or("no Gemma4 batch prepared")?;
        let engine: &Gemma4Engine = family("a gemma4 model")?;
        st().out = Value::Array(engine.finish(&prepared, slice(ptr, n))?).to_json().into_bytes();
        Ok(())
    })())
}

/// A plain byte vector the caller fills and then hands over (`kevala_kev_convert_source`).
#[no_mangle]
pub extern "C" fn kevala_alloc_vec(len: usize) -> *mut u8 {
    let mut v = std::mem::ManuallyDrop::new(vec![0u8; len]);
    v.as_mut_ptr()
}

/// Starts converting a Kev checkpoint in the browser. Inputs: the base safetensors from byte 0
/// through its header, the base config.json, Kev's tokenizer.json, adapter_model.safetensors,
/// adapter_config.json, head.pt and the provenance JSON. Allocates the pack and writes
/// `{"total", "pack", "sources": [[offset, len], ...]}` (base file byte ranges, in file order).
///
/// # Safety
/// Each pointer must be valid for reads of the length that follows it.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kevala_kev_convert_plan(
    head: *const u8,
    head_len: usize,
    cfg: *const u8,
    cfg_len: usize,
    tok: *const u8,
    tok_len: usize,
    adapter: *const u8,
    adapter_len: usize,
    acfg: *const u8,
    acfg_len: usize,
    hpt: *const u8,
    hpt_len: usize,
    model: *const u8,
    model_len: usize,
    block: usize,
) -> u32 {
    done((|| {
        let head = slice(head, head_len);
        let model = json_input(model, model_len)?;
        let mut c = kevala::kev::convert::KevConvert::new(
            head,
            input(cfg, cfg_len)?,
            input(tok, tok_len)?,
            slice(adapter, adapter_len),
            input(acfg, acfg_len)?,
            slice(hpt, hpt_len),
            block,
            model,
        )?;
        let mut pack = AlignedBuf::new(c.total);
        c.begin(pack.as_mut_slice())?;
        let sources = c.sources();
        let s = st();
        s.out = Value::Object(vec![
            ("total".into(), Value::Int(c.total.to_string())),
            ("pack".into(), Value::Int((pack.as_mut_ptr() as usize).to_string())),
            (
                "sources".into(),
                Value::Array(
                    sources
                        .iter()
                        .map(|(_, a, l)| Value::Array(vec![Value::Int(a.to_string()), Value::Int(l.to_string())]))
                        .collect(),
                ),
            ),
        ])
        .to_json()
        .into_bytes();
        s.kev_convert = Some((c, sources.into_iter().map(|x| x.0).collect()));
        s.pack = Some(pack);
        Ok(())
    })())
}

/// Hands over source `i` (a `kevala_alloc_vec` buffer) and writes every pack tensor it completes.
/// Writes `1` to the output when the pack is complete.
///
/// # Safety
/// `ptr`/`len` must come from `kevala_alloc_vec`. The caller must not use or free the buffer
/// afterwards.
#[no_mangle]
pub unsafe extern "C" fn kevala_kev_convert_source(i: usize, ptr: *mut u8, len: usize) -> u32 {
    done((|| {
        let bytes = Vec::from_raw_parts(ptr, len, len);
        let s = st();
        let (c, names) = s.kev_convert.as_mut().ok_or("no Kev conversion in progress")?;
        let name = names.get(i).ok_or("no such source")?.clone();
        let pack = s.pack.as_mut().ok_or("no Kev conversion in progress")?;
        c.add_source(&name, bytes, pack.as_mut_slice())?;
        s.out = if c.finished() { b"1".to_vec() } else { b"0".to_vec() };
        Ok(())
    })())
}

/// Selects the CPU register tile (0 = 2x4, 1 = 4x4, which only pays off with 32 vector registers).
#[no_mangle]
pub extern "C" fn kevala_set_tile(t: u32) {
    kevala::kernels::set_tile(t as u8);
}

/// A small synthetic matmul with the current tile, for the host to time.
#[no_mangle]
pub extern "C" fn kevala_tile_probe() {
    let (t, n, k) = (32, 256, 1024);
    let x = vec![0.5f32; t * k];
    let q = vec![1i8; n * k];
    let s = vec![0.01f32; n * k / 32];
    let mut out = vec![0.0; t * n];
    let mut panel = Vec::new();
    for _ in 0..2 {
        kevala::kernels::linear(
            &x,
            t,
            kevala::kernels::Mat::Q8 { n, k, block: 32, q: &q, scales: &s },
            None,
            &mut out,
            &mut panel,
        );
    }
}
