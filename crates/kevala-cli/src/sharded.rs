//! Laya's tensor-parallel trunk inside one process: what the browser's shard workers compute.

use kevala::laya::model::{build_shard, Batch, Scratch, SegOut, ShardPlan, Trunk};
use kevala::laya::Engine;

/// The trunk split `n` ways, one thread per shard and step.
pub struct Sharded {
    shards: Vec<Trunk>,
}

impl Sharded {
    pub fn new(e: &Engine, n: usize) -> Result<Sharded, String> {
        let full = e.coord.store().bytes();
        let mut shards = Vec::new();
        for i in 0..n {
            let s = build_shard(&e.cfg, full, ShardPlan { index: i, count: n })?;
            shards.push(Trunk::new(e.cfg.clone(), std::sync::Arc::new(s), i == 0)?);
        }
        Ok(Sharded { shards })
    }

    pub fn forward(&self, e: &Engine, batch: &Batch) -> Vec<SegOut> {
        let cfg = &e.cfg;
        let n = batch.tokens() * cfg.hidden;
        let mut x = e.coord.embed(batch);
        let mut scratch: Vec<Scratch> = self.shards.iter().map(|_| Scratch::default()).collect();
        let mut outs: Vec<Vec<f32>> = self.shards.iter().map(|_| vec![0.0; n]).collect();
        for s in 0..cfg.steps() {
            if s == 2 * cfg.layers {
                e.coord.bridge(&mut x, batch);
            }
            std::thread::scope(|sc| {
                for ((sh, scr), out) in self.shards.iter().zip(scratch.iter_mut()).zip(outs.iter_mut()) {
                    let x = &x;
                    sc.spawn(move || sh.step(s, x, batch, out, scr));
                }
            });
            // Sum the partials in shard order, then add the sum to the residual stream: the order
            // of the browser's reduction, so both round the same way.
            let (sum, rest) = outs.split_first_mut().expect("at least one shard");
            for out in rest.iter() {
                for (a, b) in sum.iter_mut().zip(out) {
                    *a += b;
                }
            }
            for (a, b) in x.iter_mut().zip(sum.iter()) {
                *a += b;
            }
        }
        e.coord.score(&x, batch)
    }
}
