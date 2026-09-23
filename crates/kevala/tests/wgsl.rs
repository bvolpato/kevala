//! Every GPU kernel must render for every specialization the runtimes ask for: no directive or
//! placeholder left behind, one compute entry point, and the options actually changing the
//! source where they should.

use kevala::gpu::{wgsl, KevSpec, Spec, KERNELS};

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
    out.push(Spec { row56: true, ..Default::default() });
    out.push(Spec { shape: Some((3072, 1024)), ..Default::default() });
    for hidden in [2048, 2560, 4096] {
        let kev = if hidden == 2048 {
            KevSpec { hidden, ..Default::default() }
        } else {
            KevSpec { hidden, heads: 16, kv_heads: 4, lin_heads: 32, ..Default::default() }
        };
        out.push(Spec { kev, ..Default::default() });
    }
    out
}

#[test]
fn every_kernel_renders_completely() {
    for kernel in KERNELS {
        for spec in specs() {
            if *kernel == "matmul_wide" && spec.groups != 1 {
                assert!(wgsl(kernel, &spec).is_err());
                continue;
            }
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
    for fields in [r#""f16":true"#, r#""rows":3"#, r#""groups":2"#] {
        let json = kevala::json::Value::parse(&format!(r#"{{"kernel":"matmul","row56":true,{fields}}}"#)).unwrap();
        assert_eq!(kevala::gpu::wgsl_json(&json).unwrap_err(), "row56 requires f32 rows=4 groups=1");
    }
}

#[test]
fn qwen_dimensions_validate_grouped_heads() {
    let parse = |fields: &str| Spec::from_json(&kevala::json::Value::parse(fields).unwrap());
    let s = parse(r#"{"kev":{"hidden":2560,"heads":16,"kv_heads":4,"lin_key_heads":16,"lin_heads":32,"rotary":64}}"#)
        .unwrap();
    assert_eq!(s.kev, KevSpec { hidden: 2560, heads: 16, kv_heads: 4, lin_key_heads: 16, lin_heads: 32, rotary: 64 });
    for bad in [
        r#"{"kev":{"hidden":2559}}"#,
        r#"{"kev":{"kv_heads":0}}"#,
        r#"{"kev":{"kv_heads":3}}"#,
        r#"{"kev":{"lin_key_heads":0}}"#,
        r#"{"kev":{"lin_key_heads":32,"lin_heads":16}}"#,
        r#"{"kev":{"lin_heads":256}}"#,
        r#"{"kev":{"rotary":257}}"#,
    ] {
        assert!(parse(bad).is_err(), "accepted {bad}");
    }
}
