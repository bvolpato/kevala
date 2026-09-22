//! Every GPU kernel must render for every specialization the runtimes ask for: no directive or
//! placeholder left behind, one compute entry point, and the options actually changing the
//! source where they should.

use kevala::gpu::{wgsl, Spec, KERNELS};

fn specs() -> Vec<Spec> {
    let mut out = Vec::new();
    for f16 in [false, true] {
        for subgroups in [false, true] {
            for rows in 1..=4 {
                for groups in 1..=2 {
                    out.push(Spec { f16, subgroups, rows, groups, ..Default::default() });
                }
            }
        }
    }
    out.push(Spec { shape: Some((3072, 1024)), ..Default::default() });
    out
}

#[test]
fn every_kernel_renders_completely() {
    for kernel in KERNELS {
        for spec in specs() {
            let src = wgsl(kernel, &spec).unwrap_or_else(|e| panic!("{kernel} {spec:?}: {e}"));
            assert!(!src.contains("{{") && !src.contains("//#"), "{kernel} {spec:?} left a template marker");
            assert_eq!(src.matches("@compute").count(), 1, "{kernel} {spec:?}: one entry point");
            assert!(src.contains("fn main("), "{kernel} {spec:?}: entry point is main");
        }
    }
}

#[test]
fn specializations_change_what_they_should() {
    let base = Spec::default();
    let f16 = Spec { f16: true, ..Default::default() };
    let mm = wgsl("matmul", &base).unwrap();
    let mm16 = wgsl("matmul", &f16).unwrap();
    assert!(!mm.contains("enable f16;") && mm16.starts_with("// Y[T, N]") && mm16.contains("enable f16;"));
    assert!(mm16.contains("array<vec4<f16>, 512>"));

    let two_rows = wgsl("matmul", &Spec { rows: 2, ..Default::default() }).unwrap();
    assert!(two_rows.contains("wg.y * 32u") && two_rows.contains("((T + 31u) / 32u)"));

    let shaped = wgsl("matmul", &Spec { shape: Some((3072, 1024)), ..Default::default() }).unwrap();
    assert!(shaped.contains("let N = 3072u;") && !shaped.contains("let N = p.N;"));

    // the reduce pairs with the matmul of the same tiling: same split rule
    let splits = |src: &str| src.lines().find(|l| l.contains("let tiles")).unwrap().trim().to_string();
    assert_eq!(splits(&two_rows), splits(&wgsl("reduce", &Spec { rows: 2, ..Default::default() }).unwrap()));

    let attn = wgsl("kev_attention", &base).unwrap();
    let attn_sg = wgsl("kev_attention", &Spec { subgroups: true, ..Default::default() }).unwrap();
    assert!(!attn.contains("subgroupMax") && attn_sg.contains("subgroupMax") && attn_sg.contains("enable subgroups;"));
}

#[test]
fn bad_requests_are_errors() {
    assert!(wgsl("splits", &Spec::default()).is_err(), "snippets are not kernels");
    assert!(wgsl("nope", &Spec::default()).is_err());
    let json = kevala::json::Value::parse(r#"{"kernel": "matmul", "rows": 5}"#).unwrap();
    assert!(kevala::gpu::wgsl_json(&json).is_err());
}
