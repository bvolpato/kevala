//! What every checkpoint converter shares: reading source floats, int8 quantization, and the
//! checks that a checkpoint's tokenizer and tensor names are the ones a pack can use.

use crate::json::Value;
use crate::tokenizer::Tokenizer;

/// Weight storage for decoder backbones. Norms, recurrent gates, and decision heads stay f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Bf16,
    F32,
    Q8,
}

impl Precision {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "f32" => Ok(Self::F32),
            "q8" => Ok(Self::Q8),
            _ => Err(format!("unsupported precision {value:?}; expected bf16, f32, or q8")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::Q8 => "q8",
        }
    }
}

pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

/// Round to nearest, ties to even. Values widened from BF16 retain their exact bits.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0xffff == 0 {
        return (bits >> 16) as u16;
    }
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return (bits >> 16) as u16 | 0x0040;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub(crate) fn text_tensor_prefix(has: impl Fn(&str) -> bool) -> Result<&'static str, String> {
    let mut prefixes = ["model.language_model.", "language_model.", "model."]
        .into_iter()
        .filter(|prefix| has(&format!("{prefix}embed_tokens.weight")));
    let prefix = prefixes.next().ok_or("checkpoint has no supported text embedding tensor namespace")?;
    if prefixes.next().is_some() {
        return Err("checkpoint has ambiguous text embedding tensor namespaces".into());
    }
    Ok(prefix)
}

pub(crate) fn validate_tokenizer_vocab(tokens: usize, configured: usize, embedding_rows: usize) -> Result<(), String> {
    if tokens > configured || tokens > embedding_rows {
        return Err(format!("tokenizer ID range 0..{tokens} exceeds the configured vocabulary ({configured}) or embedding rows ({embedding_rows})"));
    }
    Ok(())
}

pub(crate) fn require_prompt_tokens(tokenizer: &Tokenizer, tokens: &[&str]) -> Result<(), String> {
    for &token in tokens {
        let id = tokenizer.token_id(token).ok_or_else(|| format!("tokenizer has no required prompt token {token}"))?;
        if tokenizer.encode(token) != [id] {
            return Err(format!("required prompt token {token} does not encode to its exact token ID"));
        }
    }
    Ok(())
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, m) => sign | (0xff << 23) | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Symmetric absmax int8 per `block` columns of each row.
pub fn quantize(w: &[f32], rows: usize, cols: usize, block: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(cols % block == 0);
    let nb = cols / block;
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows * nb];
    for r in 0..rows {
        for b in 0..nb {
            let s = &w[r * cols + b * block..r * cols + (b + 1) * block];
            let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = amax / 127.0;
            scales[r * nb + b] = scale;
            if scale > 0.0 {
                let inv = 1.0 / scale;
                for (i, v) in s.iter().enumerate() {
                    q[r * cols + b * block + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (q, scales)
}

/// A JSON number for a pack config: an integer when the value is whole.
pub(crate) fn num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        Value::Int((v as i64).to_string())
    } else {
        Value::Float(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_preserves_values_and_uses_ties_to_even() {
        for bits in [0x0000, 0x8000, 0x0001, 0x3f80, 0xbf80, 0x7f7f, 0x7f80, 0xff80, 0x7f81, 0xffff] {
            assert_eq!(f32_to_bf16(bf16_to_f32(bits)), bits);
        }
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_roundtrip_edges() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // the smallest subnormal
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn quantize_is_symmetric_absmax() {
        let w = [0.5, -1.0, 0.25, 0.0, 2.0, -2.0, 1.0, 0.0];
        let (q, s) = quantize(&w, 2, 4, 4);
        assert_eq!(s, vec![1.0 / 127.0, 2.0 / 127.0]);
        assert_eq!(q, vec![64, -127, 32, 0, 127, -127, 64, 0]);
    }

    #[test]
    fn text_namespace_is_selected_from_tensors_and_rejects_ambiguity() {
        for prefix in ["model.", "model.language_model.", "language_model."] {
            let tensor = format!("{prefix}embed_tokens.weight");
            assert_eq!(text_tensor_prefix(|name| name == tensor).unwrap(), prefix);
        }
        assert!(text_tensor_prefix(|_| false).unwrap_err().contains("no supported"));
        assert!(text_tensor_prefix(|_| true).unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn tokenizer_bounds_allow_padded_embeddings_and_reject_out_of_range_ids() {
        assert!(validate_tokenizer_vocab(248077, 248320, 248320).is_ok());
        assert!(validate_tokenizer_vocab(248321, 248320, 248320).is_err());
        assert!(validate_tokenizer_vocab(248077, 248320, 248076).is_err());
    }
}
