//! kevala: a dependency-free engine for System 1 decision models (Laya, Kev, SemIf, Gemma 4), built
//! for WebAssembly and WebGPU.
//!
//! - Requests and families: [`content`] (the request model), [`runtime`] (the family registry),
//!   and one module per family: [`laya`], [`kev`] (Kev and SemIf), [`gemma4`]. Each family owns
//!   its request template, its layers, its readout, and its checkpoint converter.
//! - Weights: [`pack`] (the `.kevala` format and sub-pack layouts), [`store`] (a loaded pack),
//!   [`convert`] and [`torchpt`] (what the converters share).
//! - Text: [`json`], [`unicode`], [`tokenizer`].
//! - Numbers: [`simd`] and [`kernels`] (CPU), [`gpu`] (the WGSL kernels), [`math`].
pub mod content;
pub mod convert;
pub mod direct_options;
pub mod gemma4;
pub mod gpu;
pub mod json;
pub mod kernels;
pub mod kev;
pub mod laya;
pub mod math;
pub mod pack;
pub mod runtime;
pub mod simd;
pub mod store;
pub mod tokenizer;
pub mod torchpt;
pub mod unicode;
mod unicode_tables;
