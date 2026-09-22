//! The C ABI the JavaScript runtime calls. One instance plays one role:
//!
//! - engine: a whole pack, answers requests on its own (`kevala_engine_load` + `kevala_decide`);
//! - coordinator: the coordinator sub-pack, tokenizes and scores while shards or the GPU run the
//!   transformer layers (`kevala_prepare`, `kevala_embed`, `kevala_bridge`, `kevala_finish`);
//! - shard: one tensor-parallel slice of the layers (`kevala_shard_load`, `kevala_shard_step`).
//!
//! Strings and results come back through `kevala_out_ptr/len`, errors through `kevala_error_ptr/len`.
//! Every call that can fail returns 0 on success and 1 on error.

use kevala::content::Request;
use kevala::engine::{Engine, Prepared};
use kevala::json::Value;
use kevala::kev::KevEngine;
use kevala::model::{self, AlignedBuf, Batch, Scratch, Seg, ShardPlan, Trunk};
use kevala::runtime::{self, Model};

#[cfg(feature = "cpu-bench")]
mod cpu_bench;

#[derive(Default)]
struct State {
    model: Option<Box<dyn Model>>,
    shard: Option<Trunk>,
    prepared: Option<(Prepared, Vec<Value>)>,
    batch: Batch,
    x: Vec<f32>,
    partial: Vec<f32>,
    scratch: Scratch,
    out: Vec<u8>,
    err: String,
    plan: Option<kevala::convert::Plan>,
    pack: Option<AlignedBuf>,
    kev_batch: Option<(Vec<kevala::kev::Encoded>, Vec<Vec<kevala::kev::KevQuestion>>)>,
    kev_convert: Option<(kevala::convert_kev::KevConvert, Vec<String>)>,
    kev_ids: [Vec<u32>; 2],
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

fn input(ptr: *const u8, len: usize) -> Result<&'static str, String> {
    let b = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(b).map_err(|_| "input is not UTF-8".to_string())
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
#[no_mangle]
pub extern "C" fn kevala_layouts(ptr: *const u8, len: usize, count: usize) -> u32 {
    done((|| {
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        let h = kevala::pack::parse_header(bytes)?;
        let mut out = Vec::new();
        write_layout(&mut out, &model::coord_layout(&h)?);
        if h.config().get("arch").and_then(Value::as_str) == Some("kev") {
            // no tensor-parallel shards for Kev: the whole trunk, for the GPU
            write_layout(&mut out, &model::trunk_layout(&h)?);
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
#[no_mangle]
pub extern "C" fn kevala_engine_load(ptr: *mut u8, len: usize) -> u32 {
    done((|| {
        let m = runtime::load(unsafe { AlignedBuf::from_raw(ptr, len) })?;
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

/// The families this build knows: `[{"arch", "about"}]`.
#[no_mangle]
pub extern "C" fn kevala_families() -> u32 {
    let list = runtime::FAMILIES
        .iter()
        .map(|f| {
            Value::Object(vec![
                ("arch".into(), Value::Str(f.arch.into())),
                ("about".into(), Value::Str(f.about.into())),
            ])
        })
        .collect();
    st().out = Value::Array(list).to_json().into_bytes();
    0
}

fn requests(v: &Value) -> Result<Vec<(Value, Value)>, String> {
    Ok(Request::parse_many(v)?.into_iter().map(|r| (r.state, r.questions)).collect())
}

fn prepare_json(ptr: *const u8, len: usize) -> Result<(), String> {
    let e: &Engine = family("a laya model")?;
    let req = Value::parse(input(ptr, len)?).map_err(|e| e.to_string())?;
    let rs = requests(&req)?;
    let pairs: Vec<(&Value, &Value)> = rs.iter().map(|(s, q)| (s, q)).collect();
    let p = e.prepare(&pairs)?;
    let qs = rs.into_iter().map(|(_, q)| q).collect();
    st().prepared = Some((p, qs));
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
#[no_mangle]
pub extern "C" fn kevala_decide(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let req = Value::parse(input(ptr, len)?).map_err(|e| e.to_string())?;
        let rs = Request::parse_many(&req)?;
        let s = st();
        let out = s.model.as_mut().ok_or("no model loaded")?.decide(&rs)?;
        s.out = Value::Array(out).to_json().into_bytes();
        Ok(())
    })())
}

/// Tokenizes a request for an external trunk. Writes the segment table as u32s:
/// `tokens, segments, then per segment start, len, qtype, markers, marker positions...`.
#[no_mangle]
pub extern "C" fn kevala_prepare(ptr: *const u8, len: usize) -> u32 {
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

/// Token ids of the prepared batch.
#[no_mangle]
pub extern "C" fn kevala_ids_ptr() -> *const u32 {
    st().prepared.as_ref().map_or(std::ptr::null(), |(p, _)| p.batch.ids.as_ptr())
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
#[no_mangle]
pub extern "C" fn kevala_shard_load(ptr: *mut u8, len: usize, primary: u32) -> u32 {
    done((|| {
        let (store, h) = model::load_store(unsafe { AlignedBuf::from_raw(ptr, len) })?;
        let cfg = model::Config::from_json(h.config())?;
        st().shard = Some(Trunk::new(cfg, std::sync::Arc::new(store), primary != 0)?);
        Ok(())
    })())
}

/// Sets the batch a shard computes on from the table `kevala_prepare` wrote, and sizes the
/// residual buffer. Returns the residual buffer for the caller to fill before each step.
#[no_mangle]
pub extern "C" fn kevala_shard_batch(ptr: *const u32, len: usize) -> *mut f32 {
    let s = st();
    let t = unsafe { std::slice::from_raw_parts(ptr, len) };
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

/// Token ids for `text`, as a JSON array (debugging and tests).
/// The WGSL source of a GPU kernel specialized as asked: `{ kernel, f16, subgroups, rows, groups, n, k }`.
#[no_mangle]
pub extern "C" fn kevala_wgsl(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let request = Value::parse(input(ptr, len)?).map_err(|e| e.to_string())?;
        st().out = kevala::gpu::wgsl_json(&request)?.into_bytes();
        Ok(())
    })())
}

#[no_mangle]
pub extern "C" fn kevala_tokenize(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let tok = st().model.as_ref().ok_or("no model loaded")?.tokenizer();
        let ids = tok.encode(input(ptr, len)?);
        st().out = Value::Array(ids.into_iter().map(|i| Value::Int(i.to_string())).collect()).to_json().into_bytes();
        Ok(())
    })())
}

/// Starts an in-browser conversion. Inputs are the safetensors file from byte 0 through its JSON
/// header, and the three upstream config files. Allocates the output pack and writes a JSON job
/// list: `{"total", "pack": ptr, "jobs": [[src_offset, src_len], ...]}` in file order.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn kevala_convert_plan(
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
        let head = unsafe { std::slice::from_raw_parts(head, head_len) };
        let model = Value::parse(input(model, model_len)?).map_err(|e| e.to_string())?;
        let opt = kevala::convert::Options { block, model, keep_f32: Vec::new() };
        let plan =
            kevala::convert::plan(head, input(enc, enc_len)?, input(agent, agent_len)?, input(tok, tok_len)?, &opt)?;
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
#[no_mangle]
pub extern "C" fn kevala_convert_job(i: usize, src: *const u8, len: usize) -> u32 {
    done((|| {
        let s = st();
        let plan = s.plan.as_ref().ok_or("no conversion in progress")?;
        let job = plan.jobs.get(i).ok_or("no such job")?;
        if len != job.src_len {
            return Err(format!("job {i} ({}) needs {} bytes, got {len}", job.src, job.src_len));
        }
        let src = unsafe { std::slice::from_raw_parts(src, len) };
        kevala::convert::run_job(job, src, plan.block, s.pack.as_mut().unwrap().as_mut_slice());
        Ok(())
    })())
}

/// Hands the finished pack to the engine loader of this same instance (no copy).
#[no_mangle]
pub extern "C" fn kevala_convert_finish_load() -> u32 {
    done((|| {
        let s = st();
        s.plan = None;
        let pack = s.pack.take().ok_or("no converted pack")?;
        let m = runtime::load(pack)?;
        s.out = m.info().to_json().into_bytes();
        s.model = Some(m);
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
#[no_mangle]
pub extern "C" fn kevala_kev_prepare(ptr: *const u8, len: usize) -> u32 {
    done((|| {
        let req = Value::parse(input(ptr, len)?).map_err(|e| e.to_string())?;
        let rs = requests(&req)?;
        let e: &KevEngine = family("a kev model")?;
        let (enc, qs) = e.prepare_all(&rs)?;
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
#[no_mangle]
pub extern "C" fn kevala_kev_finish(ptr: *const f32, n: usize) -> u32 {
    done((|| {
        let e: &mut KevEngine = family("a kev model")?;
        let s = st();
        let (enc, qs) = s.kev_batch.take().ok_or("nothing prepared")?;
        let rows = unsafe { std::slice::from_raw_parts(ptr, n) };
        let logits = e.model.readout(rows, &enc);
        s.out = Value::Array(e.respond(&enc, &qs, &logits)).to_json().into_bytes();
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
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn kevala_kev_convert_plan(
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
        let bytes = |p: *const u8, n: usize| unsafe { std::slice::from_raw_parts(p, n) };
        let head = bytes(head, head_len);
        let n = u64::from_le_bytes(head.get(..8).ok_or("short safetensors")?.try_into().unwrap()) as usize;
        let model = Value::parse(input(model, model_len)?).map_err(|e| e.to_string())?;
        let mut c = kevala::convert_kev::KevConvert::new(
            head,
            input(cfg, cfg_len)?,
            input(tok, tok_len)?,
            bytes(adapter, adapter_len),
            input(acfg, acfg_len)?,
            bytes(hpt, hpt_len),
            block,
            model,
        )?;
        let mut pack = AlignedBuf::new(c.total);
        c.begin(pack.as_mut_slice())?;
        let sources = c.sources(n);
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
#[no_mangle]
pub extern "C" fn kevala_kev_convert_source(i: usize, ptr: *mut u8, len: usize) -> u32 {
    done((|| {
        let bytes = unsafe { Vec::from_raw_parts(ptr, len, len) };
        let s = st();
        let (c, names) = s.kev_convert.as_mut().ok_or("no Kev conversion in progress")?;
        let name = names.get(i).ok_or("no such source")?.clone();
        c.add_source(&name, bytes, s.pack.as_mut().unwrap().as_mut_slice())?;
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

/// Token ids of the prepared Kev batch: stage 1 (every state, concatenated) or stage 2.
#[no_mangle]
pub extern "C" fn kevala_kev_ids(stage: usize) -> *const u32 {
    match stage {
        1 | 2 => st().kev_ids[stage - 1].as_ptr(),
        _ => std::ptr::null(),
    }
}
