//! DeepSeek V4 attention-weight vindex storage (`dsv4-vindex-extraction`
//! V1, the round-trip gate).
//!
//! The generic vindex attention writer assumes a 4-tensor Q/K/V/O layout.
//! DSv4 has none of those — it has **low-rank Q** (`attn_q_a`/`attn_q_b`),
//! a **latent KV** (`attn_kv_latent`), and a **grouped low-rank output**
//! projection (`attn_output_a`/`attn_output_b`). This module serializes
//! those five weights — as raw quantized bytes (Q4_K/Q6_K passthrough, no
//! dequant/recompress, mirroring the resident loader) — plus the small
//! f32 norms and per-head sinks, to the `dsv4_attn.bin` wire format, and
//! reads them back losslessly. A DSv4 vindex reader (and the round-trip
//! test below) reconstruct the attention half of `DsV4LayerWeightStorage`
//! from this.
//!
//! Hand-rolled little-endian, no serde — same style as
//! [`super::dsv4_kv_persist`]. All reads are bounds-checked.

use super::dsv4_gguf_reader::RawExpertTensor;

/// 4-byte magic: "D4VA" (DSv4 Vindex Attention).
const MAGIC: [u8; 4] = *b"D4VA";
const VERSION: u16 = 1;

/// One DSv4 layer's attention weights in extraction form: the five raw
/// quantized projection tensors + the f32 norms/sinks.
#[derive(Clone)]
pub struct DsV4AttnWeights {
    /// `(q_lora_rank, n_embd)` Q-down.
    pub q_a: RawExpertTensor,
    /// `(n_head*head_dim, q_lora_rank)` Q-up.
    pub q_b: RawExpertTensor,
    /// `(head_dim, n_embd)` latent KV-down.
    pub kv_latent: RawExpertTensor,
    /// `(n_groups*o_lora_rank, group_dim)` grouped O-a.
    pub output_a: RawExpertTensor,
    /// `(n_embd, o_lora_rank*n_groups)` shared O-b.
    pub output_b: RawExpertTensor,
    /// `(n_embd,)` input RMSNorm gain.
    pub attn_norm: Vec<f32>,
    /// `(q_lora_rank,)` RMSNorm gain on the q_a output.
    pub q_a_norm: Vec<f32>,
    /// `(head_dim,)` RMSNorm gain on the kv output.
    pub kv_a_norm: Vec<f32>,
    /// Optional `(n_head,)` per-head attention sinks.
    pub attn_sinks: Option<Vec<f32>>,
}

/// Error decoding a `dsv4_attn.bin` blob.
#[derive(Debug, PartialEq, Eq)]
pub enum DsV4AttnPersistError {
    BadMagic([u8; 4]),
    UnsupportedVersion(u16),
    Truncated {
        what: &'static str,
        need: usize,
        have: usize,
    },
}

impl std::fmt::Display for DsV4AttnPersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4AttnPersistError::BadMagic(m) => write!(f, "bad magic {m:?} (expected {MAGIC:?})"),
            DsV4AttnPersistError::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            DsV4AttnPersistError::Truncated { what, need, have } => {
                write!(f, "truncated reading {what}: need {need}, have {have}")
            }
        }
    }
}
impl std::error::Error for DsV4AttnPersistError {}

/// Serialize one layer's attention weights to a `dsv4_attn.bin` blob.
pub fn serialize_dsv4_attn(w: &DsV4AttnWeights) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    // Five raw tensors, fixed order (no tags needed).
    for t in [&w.q_a, &w.q_b, &w.kv_latent, &w.output_a, &w.output_b] {
        put_raw(&mut out, t);
    }
    put_f32_vec(&mut out, &w.attn_norm);
    put_f32_vec(&mut out, &w.q_a_norm);
    put_f32_vec(&mut out, &w.kv_a_norm);
    match &w.attn_sinks {
        Some(s) => {
            out.push(1);
            put_f32_vec(&mut out, s);
        }
        None => out.push(0),
    }
    out
}

/// Deserialize one layer's attention weights from a `dsv4_attn.bin` blob.
pub fn deserialize_dsv4_attn(bytes: &[u8]) -> Result<DsV4AttnWeights, DsV4AttnPersistError> {
    let mut c = Cursor::new(bytes);
    let magic = c.take_array4("magic")?;
    if magic != MAGIC {
        return Err(DsV4AttnPersistError::BadMagic(magic));
    }
    let version = c.read_u16("version")?;
    if version != VERSION {
        return Err(DsV4AttnPersistError::UnsupportedVersion(version));
    }
    let q_a = c.read_raw("q_a")?;
    let q_b = c.read_raw("q_b")?;
    let kv_latent = c.read_raw("kv_latent")?;
    let output_a = c.read_raw("output_a")?;
    let output_b = c.read_raw("output_b")?;
    let attn_norm = c.read_f32_vec("attn_norm")?;
    let q_a_norm = c.read_f32_vec("q_a_norm")?;
    let kv_a_norm = c.read_f32_vec("kv_a_norm")?;
    let attn_sinks = if c.read_u8("has_sinks")? != 0 {
        Some(c.read_f32_vec("attn_sinks")?)
    } else {
        None
    };
    Ok(DsV4AttnWeights {
        q_a,
        q_b,
        kv_latent,
        output_a,
        output_b,
        attn_norm,
        q_a_norm,
        kv_a_norm,
        attn_sinks,
    })
}

// ── encode helpers ──────────────────────────────────────────────────────

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `tensor_type:u32 | rows:u32 | cols:u32 | byte_len:u64 | bytes`.
fn put_raw(out: &mut Vec<u8>, t: &RawExpertTensor) {
    put_u32(out, t.tensor_type);
    put_u32(out, t.rows as u32);
    put_u32(out, t.cols as u32);
    out.extend_from_slice(&(t.bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&t.bytes);
}

/// `len:u32 | len*f32`.
fn put_f32_vec(out: &mut Vec<u8>, v: &[f32]) {
    put_u32(out, v.len() as u32);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

// ── decode helpers ──────────────────────────────────────────────────────

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], DsV4AttnPersistError> {
        let have = self.buf.len().saturating_sub(self.pos);
        if n > have {
            return Err(DsV4AttnPersistError::Truncated {
                what,
                need: n,
                have,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn take_array4(&mut self, what: &'static str) -> Result<[u8; 4], DsV4AttnPersistError> {
        let s = self.take(4, what)?;
        Ok([s[0], s[1], s[2], s[3]])
    }
    fn read_u8(&mut self, what: &'static str) -> Result<u8, DsV4AttnPersistError> {
        Ok(self.take(1, what)?[0])
    }
    fn read_u16(&mut self, what: &'static str) -> Result<u16, DsV4AttnPersistError> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn read_u32(&mut self, what: &'static str) -> Result<u32, DsV4AttnPersistError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn read_u64(&mut self, what: &'static str) -> Result<u64, DsV4AttnPersistError> {
        let s = self.take(8, what)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    fn read_raw(&mut self, what: &'static str) -> Result<RawExpertTensor, DsV4AttnPersistError> {
        let tensor_type = self.read_u32(what)?;
        let rows = self.read_u32(what)? as usize;
        let cols = self.read_u32(what)? as usize;
        let byte_len = self.read_u64(what)? as usize;
        let bytes = self.take(byte_len, what)?.to_vec();
        Ok(RawExpertTensor {
            bytes,
            tensor_type,
            rows,
            cols,
        })
    }

    fn read_f32_vec(&mut self, what: &'static str) -> Result<Vec<f32>, DsV4AttnPersistError> {
        let n = self.read_u32(what)? as usize;
        let s = self.take(n * 4, what)?;
        let mut v = Vec::with_capacity(n);
        for chunk in s.chunks_exact(4) {
            v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;

    fn raw(rows: usize, cols: usize, seed: u8) -> RawExpertTensor {
        RawExpertTensor {
            bytes: (0..rows * cols)
                .flat_map(|i| ((i as f32) * 0.001 + seed as f32).to_le_bytes())
                .collect(),
            tensor_type: TYPE_F32,
            rows,
            cols,
        }
    }

    fn sample(sinks: bool) -> DsV4AttnWeights {
        DsV4AttnWeights {
            q_a: raw(16, 64, 1),
            q_b: raw(256, 16, 2),
            kv_latent: raw(8, 64, 3),
            output_a: raw(16, 32, 4),
            output_b: raw(64, 16, 5),
            attn_norm: vec![1.0; 64],
            q_a_norm: vec![0.5; 16],
            kv_a_norm: vec![0.25; 8],
            attn_sinks: sinks.then(|| vec![-1.0; 4]),
        }
    }

    fn assert_raw_eq(a: &RawExpertTensor, b: &RawExpertTensor) {
        assert_eq!(a.tensor_type, b.tensor_type);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.bytes, b.bytes, "raw bytes differ");
    }

    /// The five attention weights + norms + sinks round-trip bit-exactly
    /// (Q4_K passthrough means the on-disk bytes equal the GGUF bytes).
    #[test]
    fn attn_weights_round_trip_losslessly() {
        let w = sample(true);
        let blob = serialize_dsv4_attn(&w);
        let got = deserialize_dsv4_attn(&blob).unwrap();
        assert_raw_eq(&got.q_a, &w.q_a);
        assert_raw_eq(&got.q_b, &w.q_b);
        assert_raw_eq(&got.kv_latent, &w.kv_latent);
        assert_raw_eq(&got.output_a, &w.output_a);
        assert_raw_eq(&got.output_b, &w.output_b);
        assert_eq!(got.attn_norm, w.attn_norm);
        assert_eq!(got.q_a_norm, w.q_a_norm);
        assert_eq!(got.kv_a_norm, w.kv_a_norm);
        assert_eq!(got.attn_sinks, w.attn_sinks);
    }

    /// Sinks absent round-trips to None.
    #[test]
    fn no_sinks_round_trips() {
        let w = sample(false);
        let got = deserialize_dsv4_attn(&serialize_dsv4_attn(&w)).unwrap();
        assert!(got.attn_sinks.is_none());
        assert_raw_eq(&got.q_b, &w.q_b);
    }

    /// The reconstructed raw tensors build `QuantTensor`s of the expected
    /// shape — i.e. the round-tripped bytes feed the resident path.
    #[test]
    fn round_tripped_raw_builds_quant_tensor() {
        use larql_models::quant::lazy::QuantTensor;
        let w = sample(false);
        let got = deserialize_dsv4_attn(&serialize_dsv4_attn(&w)).unwrap();
        let qt = QuantTensor::from_raw(
            got.q_b.bytes,
            got.q_b.tensor_type,
            got.q_b.rows,
            got.q_b.cols,
        )
        .expect("from_raw on round-tripped q_b");
        assert_eq!(qt.shape(), [256, 16]);
    }

    /// Bad magic / version / truncation → typed error, never a panic.
    #[test]
    fn malformed_blobs_are_typed_errors() {
        let w = sample(true);
        let good = serialize_dsv4_attn(&w);

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            deserialize_dsv4_attn(&bad_magic),
            Err(DsV4AttnPersistError::BadMagic(_))
        ));

        let mut bad_ver = good.clone();
        bad_ver[4] = 0xFF;
        bad_ver[5] = 0xFF;
        assert!(matches!(
            deserialize_dsv4_attn(&bad_ver),
            Err(DsV4AttnPersistError::UnsupportedVersion(0xFFFF))
        ));

        for cut in 0..good.len() {
            assert!(
                deserialize_dsv4_attn(&good[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    /// **The V1 re-evaluation gate** (dsv4-vindex-extraction task 2.3).
    ///
    /// Read layer 0's five attention weights + norms/sinks from the real
    /// DSv4-Flash GGUF — using the *same* reader the resident loader feeds
    /// — serialize to the `dsv4_attn.bin` wire format, read it back, and
    /// assert the attention half of `DsV4LayerWeightStorage` reconstructs
    /// bit-for-bit: every quantized tensor's bytes/type/shape survive the
    /// round-trip (Q4_K passthrough — no dequant), the norms/sinks match,
    /// and the restored bytes feed `QuantTensor::from_raw` at the exact
    /// `hp`-derived shapes the resident path expects.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_attn_round_trips_to_storage() {
        use super::super::dsv4_gguf_reader::{
            read_dsv4_layer_raw_expert_tensors_from_gguf, read_dsv4_layer_tensors_from_gguf,
        };
        use super::super::dsv4_storage_build::DsV4Hyperparams;
        use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
        use larql_models::loading::gguf::GgufFile;
        use larql_models::quant::ggml::TYPE_F32;
        use larql_models::quant::lazy::QuantTensor;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");

        // Same five attn kinds the resident loader reads raw.
        let attn_kinds = [
            DsV4TensorKind::AttnQA,
            DsV4TensorKind::AttnQB,
            DsV4TensorKind::AttnKv,
            DsV4TensorKind::AttnOutA,
            DsV4TensorKind::AttnOutB,
        ];
        let mut raw = read_dsv4_layer_raw_expert_tensors_from_gguf(&gguf, 0, &attn_kinds)
            .expect("read raw attn tensors");
        let f32s = read_dsv4_layer_tensors_from_gguf(&gguf, 0).expect("read f32 tensors");

        let take = |raw: &mut std::collections::HashMap<DsV4TensorKind, RawExpertTensor>, k| {
            raw.remove(&k).unwrap_or_else(|| panic!("missing {k:?}"))
        };
        let original = DsV4AttnWeights {
            q_a: take(&mut raw, DsV4TensorKind::AttnQA),
            q_b: take(&mut raw, DsV4TensorKind::AttnQB),
            kv_latent: take(&mut raw, DsV4TensorKind::AttnKv),
            output_a: take(&mut raw, DsV4TensorKind::AttnOutA),
            output_b: take(&mut raw, DsV4TensorKind::AttnOutB),
            attn_norm: f32s[&DsV4TensorKind::AttnNorm].clone(),
            q_a_norm: f32s[&DsV4TensorKind::AttnQANorm].clone(),
            kv_a_norm: f32s[&DsV4TensorKind::AttnKvANorm].clone(),
            attn_sinks: f32s.get(&DsV4TensorKind::AttnSinks).cloned(),
        };

        // The GGUF carries these attn weights quantized — the passthrough
        // must not silently degrade to f32.
        assert_ne!(
            original.q_a.tensor_type, TYPE_F32,
            "q_a should be quantized"
        );
        assert_ne!(
            original.q_b.tensor_type, TYPE_F32,
            "q_b should be quantized"
        );

        let restored = deserialize_dsv4_attn(&serialize_dsv4_attn(&original)).unwrap();

        assert_raw_eq(&restored.q_a, &original.q_a);
        assert_raw_eq(&restored.q_b, &original.q_b);
        assert_raw_eq(&restored.kv_latent, &original.kv_latent);
        assert_raw_eq(&restored.output_a, &original.output_a);
        assert_raw_eq(&restored.output_b, &original.output_b);
        assert_eq!(restored.attn_norm, original.attn_norm);
        assert_eq!(restored.q_a_norm, original.q_a_norm);
        assert_eq!(restored.kv_a_norm, original.kv_a_norm);
        assert_eq!(restored.attn_sinks, original.attn_sinks);

        // The restored bytes reconstruct the attention half of the layer
        // storage at the resident shapes. q_a/q_b/kv_latent are 2D linears
        // (`[out, in]`).
        let q_a = QuantTensor::from_raw(
            restored.q_a.bytes,
            restored.q_a.tensor_type,
            restored.q_a.rows,
            restored.q_a.cols,
        )
        .expect("from_raw q_a");
        assert_eq!(q_a.shape(), [hp.q_lora_rank, hp.n_embd]);

        let q_b = QuantTensor::from_raw(
            restored.q_b.bytes,
            restored.q_b.tensor_type,
            restored.q_b.rows,
            restored.q_b.cols,
        )
        .expect("from_raw q_b");
        assert_eq!(q_b.shape(), [hp.n_head * hp.head_dim, hp.q_lora_rank]);

        let kv = QuantTensor::from_raw(
            restored.kv_latent.bytes,
            restored.kv_latent.tensor_type,
            restored.kv_latent.rows,
            restored.kv_latent.cols,
        )
        .expect("from_raw kv_latent");
        assert_eq!(kv.shape(), [hp.head_dim, hp.n_embd]);

        // Norm/sink shapes match the storage contract.
        assert_eq!(restored.attn_norm.len(), hp.n_embd);
        assert_eq!(restored.q_a_norm.len(), hp.q_lora_rank);
        assert_eq!(restored.kv_a_norm.len(), hp.head_dim);
        if let Some(sinks) = &restored.attn_sinks {
            assert_eq!(sinks.len(), hp.n_head);
        }

        eprintln!(
            "round-trip OK: q_a/q_b/kv/o_a/o_b type {} bytes {}+{}+{}+{}+{}",
            original.q_a.tensor_type,
            original.q_a.bytes.len(),
            original.q_b.bytes.len(),
            original.kv_latent.bytes.len(),
            original.output_a.bytes.len(),
            original.output_b.bytes.len(),
        );
    }
}
