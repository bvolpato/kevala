//! Every GPU kernel must render for every specialization the runtimes ask for: no directive or
//! placeholder left behind, one compute entry point, and the options actually changing the
//! source where they should.

use kevala::gpu::{wgsl, KevSpec, Spec, WeightType, KERNELS};

fn vision_qkv_spec() -> Spec {
    Spec {
        weights: WeightType::Bf16,
        f16: false,
        rows: 4,
        row56: true,
        groups: 1,
        split_target: 256,
        ..Default::default()
    }
}

fn grid_mlp_spec() -> Spec {
    Spec {
        weights: WeightType::Q8,
        f16: false,
        rows: 4,
        row56: true,
        groups: 1,
        split_target: 256,
        ..Default::default()
    }
}

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
    out.push(Spec { rows: 2, split_target: 256, ..Default::default() });
    out.push(Spec { f16: true, rows: 4, split_target: 256, ..Default::default() });
    out.push(vision_qkv_spec());
    out.push(grid_mlp_spec());
    for weights in [WeightType::Bf16, WeightType::F32] {
        for rows in 1..=4 {
            out.push(Spec { weights, rows, ..Default::default() });
        }
    }
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
            if *kernel == "gemma4_vision_qkv" && spec != vision_qkv_spec() {
                assert_eq!(
                    wgsl(kernel, &spec).unwrap_err(),
                    "gemma4_vision_qkv requires bf16 f32 rows=4 row56=true groups=1 subgroups=false splitTarget=256 dynamic shape"
                );
                continue;
            }
            if *kernel == "gemma4_mlp_grid" && spec != grid_mlp_spec() {
                assert_eq!(
                    wgsl(kernel, &spec).unwrap_err(),
                    "gemma4_mlp_grid requires q8 f32 rows=4 row56=true groups=1 subgroups=false splitTarget=256 dynamic shape"
                );
                continue;
            }
            if *kernel == "matmul_gelu_tall"
                && (spec.weights != WeightType::Q8
                    || !spec.f16
                    || spec.rows != 4
                    || spec.groups != 1
                    || spec.row56
                    || spec.split_target != 256)
            {
                assert_eq!(
                    wgsl(kernel, &spec).unwrap_err(),
                    "matmul_gelu_tall requires q8 f16 rows=4 groups=1 row56=false splitTarget=256"
                );
                continue;
            }
            if *kernel == "matmul_tall"
                && (spec.weights != WeightType::Q8
                    || !spec.f16
                    || spec.rows != 4
                    || spec.groups != 1
                    || spec.row56
                    || spec.split_target != 256)
            {
                assert_eq!(
                    wgsl(kernel, &spec).unwrap_err(),
                    "matmul_tall requires q8 f16 rows=4 groups=1 row56=false splitTarget=256"
                );
                continue;
            }
            if matches!(*kernel, "matmul_bn32_gemma_source" | "matmul_gelu_bn32")
                && (spec.weights != WeightType::Q8
                    || spec.f16
                    || spec.rows != 2
                    || spec.groups != 1
                    || spec.row56
                    || spec.split_target != 256)
            {
                assert_eq!(
                    wgsl(kernel, &spec).unwrap_err(),
                    format!("{kernel} requires q8 f32 rows=2 groups=1 row56=false splitTarget=256")
                );
                continue;
            }
            if matches!(
                *kernel,
                "matmul_wide"
                    | "matmul_k64"
                    | "matmul_wide_gemma_batch"
                    | "matmul_wide_gemma_source"
                    | "matmul_k64_gemma_source"
                    | "matmul_gelu"
                    | "reduce_gelu"
            ) && spec.groups != 1
            {
                assert!(wgsl(kernel, &spec).is_err());
                continue;
            }
            if matches!(*kernel, "matmul_gelu" | "matvec_two" | "matvec_gelu_two") && spec.weights != WeightType::Q8 {
                assert_eq!(wgsl(kernel, &spec).unwrap_err(), format!("{kernel} requires q8 weights"));
                continue;
            }
            if *kernel == "gemma4_vision_position" && spec.weights == WeightType::Q8 {
                assert_eq!(wgsl(kernel, &spec).unwrap_err(), "gemma4_vision_position requires bf16 or f32 weights");
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
fn vision_qkv_requires_the_registered_abi_and_exports_by_name() {
    let valid = vision_qkv_spec();
    let source = wgsl("gemma4_vision_qkv", &valid).unwrap();

    let request = kevala::json::Value::parse(
        r#"{"kernel":"gemma4_vision_qkv","weights":"bf16","rows":4,"row56":true,"groups":1,"splitTarget":256}"#,
    )
    .unwrap();
    assert_eq!(kevala::gpu::wgsl_json(&request).unwrap(), source);

    let rejected = [
        ("q8", Spec { weights: WeightType::Q8, ..valid.clone() }),
        ("f32 weights", Spec { weights: WeightType::F32, ..valid.clone() }),
        ("f16", Spec { f16: true, ..valid.clone() }),
        ("subgroups", Spec { subgroups: true, ..valid.clone() }),
        ("rows", Spec { rows: 3, ..valid.clone() }),
        ("row56", Spec { row56: false, ..valid.clone() }),
        ("groups", Spec { groups: 2, ..valid.clone() }),
        ("split target", Spec { split_target: 128, ..valid.clone() }),
        ("shape", Spec { shape: Some((768, 768)), ..valid.clone() }),
        ("Qwen dimensions", Spec { kev: KevSpec { hidden: 2048, ..KevSpec::default() }, ..valid.clone() }),
    ];
    for (field, spec) in rejected {
        assert_eq!(
            wgsl("gemma4_vision_qkv", &spec).unwrap_err(),
            "gemma4_vision_qkv requires bf16 f32 rows=4 row56=true groups=1 subgroups=false splitTarget=256 dynamic shape",
            "accepted unsupported {field} specialization"
        );
    }
}

#[test]
fn grid_mlp_requires_the_registered_abi_and_exports_by_name() {
    let valid = grid_mlp_spec();
    let source = wgsl("gemma4_mlp_grid", &valid).unwrap();

    let request = kevala::json::Value::parse(
        r#"{"kernel":"gemma4_mlp_grid","weights":"q8","f16":false,"rows":4,"row56":true,"groups":1,"splitTarget":256}"#,
    )
    .unwrap();
    assert_eq!(kevala::gpu::wgsl_json(&request).unwrap(), source);

    let rejected = [
        ("bf16 weights", Spec { weights: WeightType::Bf16, ..valid.clone() }),
        ("f32 weights", Spec { weights: WeightType::F32, ..valid.clone() }),
        ("f16", Spec { f16: true, ..valid.clone() }),
        ("subgroups", Spec { subgroups: true, ..valid.clone() }),
        ("rows", Spec { rows: 3, ..valid.clone() }),
        ("row56", Spec { row56: false, ..valid.clone() }),
        ("groups", Spec { groups: 2, ..valid.clone() }),
        ("split target", Spec { split_target: 128, ..valid.clone() }),
        ("shape", Spec { shape: Some((6144, 1536)), ..valid.clone() }),
        ("Kev dimensions", Spec { kev: KevSpec { hidden: 2048, ..KevSpec::default() }, ..valid.clone() }),
    ];
    for (field, spec) in rejected {
        assert_eq!(
            wgsl("gemma4_mlp_grid", &spec).unwrap_err(),
            "gemma4_mlp_grid requires q8 f32 rows=4 row56=true groups=1 subgroups=false splitTarget=256 dynamic shape",
            "accepted unsupported {field} specialization"
        );
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

    let tall_spec = Spec { f16: true, rows: 4, split_target: 256, ..Default::default() };
    let tall = wgsl("matmul_tall", &tall_spec).unwrap();
    assert!(tall.contains("@compute @workgroup_size(32, 8)"));
    assert!(tall.contains("var<workgroup> xs: array<vec4<f16>, 832>;"));
    assert!(tall.contains("let m0 = wg.y * 104u;"));
    assert!(tall.contains("let tiles = ((N + 63u) / 64u);"));
    assert!(tall.contains("if (splits > 1u) { PART[z * T * N + o] = value; return; }"));
    assert!(tall.contains("if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }"));
    assert_eq!(tall.matches("workgroupBarrier();").count(), 2);
}

#[test]
fn bad_requests_are_errors() {
    assert!(wgsl("splits", &Spec::default()).is_err(), "snippets are not kernels");
    assert!(wgsl("nope", &Spec::default()).is_err());
    let json = kevala::json::Value::parse(r#"{"kernel": "matmul", "rows": 5}"#).unwrap();
    assert!(kevala::gpu::wgsl_json(&json).is_err());
    let json = kevala::json::Value::parse(r#"{"kernel":"matmul","weights":"f16"}"#).unwrap();
    assert_eq!(kevala::gpu::wgsl_json(&json).unwrap_err(), "weights must be q8, bf16, or f32");
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
