//! Reads the metadata Regent needs from a GGUF header, without loading the
//! weights.
//!
//! Two rules keep this safe on untrusted files:
//! - every length and count is checked against the file size before it is
//!   used, so a corrupt header can't trigger a huge allocation;
//! - values we don't need are skipped with `seek`, never read into memory.
//!
//! The reader stops early once the architecture is known and a `tokenizer.`
//! key appears: converters write tokenizer arrays last, and they are huge.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The four bytes that start every GGUF file.
const MAGIC: &[u8; 4] = b"GGUF";
/// The read buffer, in bytes.
const READ_BUFFER: usize = 64 * 1024;
/// The most metadata entries a file may declare.
const MAX_KV_COUNT: u64 = 1_000_000;
/// The longest key, in UTF-8 bytes.
const MAX_KEY_LEN: u64 = 65_536;
/// The longest architecture name, in UTF-8 bytes.
const MAX_ARCH_LEN: u64 = 256;
/// How many pre-architecture values are remembered.
const MAX_PENDING: usize = 64;
/// The largest integer array whose maximum is used for a wanted key.
const MAX_ARRAY_ELEMENTS: u64 = 4096;
/// The deepest array nesting that is accepted.
const MAX_ARRAY_DEPTH: u32 = 3;

const TY_U8: u32 = 0;
const TY_I8: u32 = 1;
const TY_U16: u32 = 2;
const TY_I16: u32 = 3;
const TY_U32: u32 = 4;
const TY_I32: u32 = 5;
const TY_F32: u32 = 6;
const TY_BOOL: u32 = 7;
const TY_STRING: u32 = 8;
const TY_ARRAY: u32 = 9;
const TY_U64: u32 = 10;
const TY_I64: u32 = 11;
const TY_F64: u32 = 12;

/// The key suffixes we care about, relative to the architecture name.
const SUFFIXES: [&str; 8] = [
    "context_length",
    "block_count",
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.sliding_window",
];

/// The model facts the memory forecast needs.
///
/// Every field is optional: a file that doesn't declare it reads as unknown,
/// never as zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ModelMeta {
    /// The model family, such as `llama` or `qwen2`.
    pub architecture: Option<String>,
    /// The trained context length, in tokens.
    pub context_length: Option<u64>,
    /// The number of layers.
    pub block_count: Option<u64>,
    /// The hidden width.
    pub embedding_length: Option<u64>,
    /// The number of attention heads.
    pub head_count: Option<u64>,
    /// The number of key/value heads.
    pub head_count_kv: Option<u64>,
    /// The key head dimension.
    pub key_length: Option<u64>,
    /// The value head dimension.
    pub value_length: Option<u64>,
    /// The sliding-window size, in tokens.
    pub sliding_window: Option<u64>,
}

impl ModelMeta {
    /// The number of KV heads, falling back to the attention head count.
    pub fn kv_heads(&self) -> Option<u64> {
        self.head_count_kv.or(self.head_count)
    }

    /// The key head dimension, falling back to width ÷ heads.
    pub fn key_dim(&self) -> Option<u64> {
        self.key_length.or_else(|| self.fallback_dim())
    }

    /// The value head dimension, falling back to width ÷ heads.
    pub fn value_dim(&self) -> Option<u64> {
        self.value_length.or_else(|| self.fallback_dim())
    }

    fn fallback_dim(&self) -> Option<u64> {
        let heads = self.head_count?;
        if heads == 0 {
            return None;
        }
        let width = self.embedding_length.unwrap_or(0);
        Some(width / heads)
    }
}

/// Why a GGUF header couldn't be read.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The file couldn't be read.
    #[error("Couldn't read the model file: {0}")]
    Io(#[from] std::io::Error),
    /// The file doesn't start with the GGUF magic.
    #[error("This file isn't a GGUF model.")]
    NotGguf,
    /// The GGUF version isn't one we understand.
    #[error("GGUF version {0} isn't supported.")]
    UnsupportedVersion(u32),
    /// The header is malformed.
    #[error("The model file is damaged: {0}")]
    Corrupt(String),
}

/// A number captured from a wanted key, in any integer form.
enum Captured {
    /// A single integer; negative values never get here.
    Int(u64),
    /// A non-negative integer array.
    IntArray(Vec<u64>),
}

/// Reads a model's metadata from a GGUF file on disk.
pub fn read_gguf_meta(path: &Path) -> Result<ModelMeta, GgufError> {
    let file = File::open(path)?;
    let total_len = file.metadata()?.len();
    read_gguf_meta_from(file, total_len)
}

/// Reads a model's metadata from any seekable byte stream.
///
/// `total_len` is the length of the whole stream; every declared length and
/// count is checked against it.
pub fn read_gguf_meta_from<R: Read + Seek>(
    reader: R,
    total_len: u64,
) -> Result<ModelMeta, GgufError> {
    let mut r = BufReader::with_capacity(READ_BUFFER, reader);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(GgufError::NotGguf);
    }
    let version = read_u32(&mut r)?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > MAX_KV_COUNT {
        return Err(GgufError::Corrupt(format!(
            "{kv_count} metadata entries is too many"
        )));
    }

    let mut meta = ModelMeta::default();
    let mut pending: Vec<(usize, Captured)> = Vec::new();

    for _ in 0..kv_count {
        let key = read_bounded_string(&mut r, total_len, MAX_KEY_LEN)?;

        // Tokenizer arrays come last and are enormous; once the architecture
        // is known every wanted key has already been seen.
        if meta.architecture.is_some() && key.starts_with("tokenizer.") {
            break;
        }

        let value_type = read_u32(&mut r)?;

        if key == "general.architecture" {
            if value_type == TY_STRING {
                let arch = read_bounded_string(&mut r, total_len, MAX_ARCH_LEN)?;
                if !arch.is_empty() {
                    meta.architecture = Some(arch);
                    for (index, value) in pending.drain(..) {
                        set_field(&mut meta, index, value);
                    }
                }
            } else {
                consume_value(&mut r, total_len, value_type, false, 1)?;
            }
            continue;
        }

        match matching_suffix(&key) {
            Some(index) if wanted_key(&meta, index, &key) => {
                if meta.architecture.is_some() {
                    if let Some(value) = consume_value(&mut r, total_len, value_type, true, 1)? {
                        set_field(&mut meta, index, value);
                    }
                } else if pending.len() < MAX_PENDING {
                    if let Some(value) = consume_value(&mut r, total_len, value_type, true, 1)? {
                        pending.push((index, value));
                    }
                } else {
                    consume_value(&mut r, total_len, value_type, false, 1)?;
                }
            }
            _ => {
                consume_value(&mut r, total_len, value_type, false, 1)?;
            }
        }
    }

    Ok(meta)
}

/// Whether `key` belongs to the architecture's field at `index`.
fn wanted_key(meta: &ModelMeta, index: usize, key: &str) -> bool {
    match &meta.architecture {
        Some(arch) => key == format!("{arch}.{}", SUFFIXES[index]),
        None => true,
    }
}

/// The wanted suffix `key` ends with, as an index into [`SUFFIXES`].
fn matching_suffix(key: &str) -> Option<usize> {
    let bytes = key.as_bytes();
    SUFFIXES.iter().position(|suffix| {
        let len = suffix.len();
        key.len() > len && key.ends_with(suffix) && bytes[key.len() - len - 1] == b'.'
    })
}

/// Stores a captured value in the field it belongs to.
fn set_field(meta: &mut ModelMeta, index: usize, value: Captured) {
    let number = match value {
        Captured::Int(value) => Some(value),
        Captured::IntArray(values) => values.into_iter().max(),
    };
    let Some(number) = number else {
        return;
    };
    match index {
        0 => meta.context_length = Some(number),
        1 => meta.block_count = Some(number),
        2 => meta.embedding_length = Some(number),
        3 => meta.head_count = Some(number),
        4 => meta.head_count_kv = Some(number),
        5 => meta.key_length = Some(number),
        6 => meta.value_length = Some(number),
        7 => meta.sliding_window = Some(number),
        _ => {}
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// The end offset of a value of `len` bytes at `pos`, if it fits the file.
fn checked_end(pos: u64, len: u64, total_len: u64) -> Result<u64, GgufError> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| GgufError::Corrupt("a length overflows".to_string()))?;
    if end > total_len {
        return Err(GgufError::Corrupt(
            "a length runs past the end of the file".to_string(),
        ));
    }
    Ok(end)
}

fn skip_bytes<R: Read + Seek>(
    r: &mut BufReader<R>,
    total_len: u64,
    len: u64,
) -> Result<(), GgufError> {
    let pos = r.stream_position()?;
    let end = checked_end(pos, len, total_len)?;
    r.seek(SeekFrom::Start(end))?;
    Ok(())
}

/// Reads a `u64`-length-prefixed string, refusing lengths above `max_len`.
fn read_bounded_string<R: Read + Seek>(
    r: &mut BufReader<R>,
    total_len: u64,
    max_len: u64,
) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    if len > max_len {
        return Err(GgufError::Corrupt(format!(
            "a string of {len} bytes is too long"
        )));
    }
    let pos = r.stream_position()?;
    checked_end(pos, len, total_len)?;
    let mut bytes = vec![0u8; len as usize];
    r.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Steps over a `u64`-length-prefixed string without reading its bytes.
fn skip_string<R: Read + Seek>(r: &mut BufReader<R>, total_len: u64) -> Result<(), GgufError> {
    let len = read_u64(r)?;
    skip_bytes(r, total_len, len)
}

/// The byte width of a fixed-size value type, if it has one.
fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        TY_U8 | TY_I8 | TY_BOOL => Some(1),
        TY_U16 | TY_I16 => Some(2),
        TY_U32 | TY_I32 | TY_F32 => Some(4),
        TY_U64 | TY_I64 | TY_F64 => Some(8),
        _ => None,
    }
}

/// The smallest number of bytes one element of `ty` can occupy.
fn elem_min_size(ty: u32) -> Option<u64> {
    match ty {
        TY_STRING => Some(8),
        TY_ARRAY => Some(12),
        other => fixed_size(other),
    }
}

fn is_int(ty: u32) -> bool {
    matches!(
        ty,
        TY_U8 | TY_I8 | TY_U16 | TY_I16 | TY_U32 | TY_I32 | TY_U64 | TY_I64
    )
}

/// Reads one integer value; negative values read as unknown.
fn read_int<R: Read>(r: &mut BufReader<R>, ty: u32) -> Result<Option<u64>, GgufError> {
    let size = fixed_size(ty)
        .ok_or_else(|| GgufError::Corrupt(format!("value type {ty} isn't an integer")))?;
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes[..size as usize])?;
    let value = match ty {
        TY_U8 => Some(u64::from(bytes[0])),
        TY_I8 => non_negative(i64::from(bytes[0] as i8)),
        TY_U16 => Some(u64::from(u16::from_le_bytes([bytes[0], bytes[1]]))),
        TY_I16 => non_negative(i64::from(i16::from_le_bytes([bytes[0], bytes[1]]))),
        TY_U32 => Some(u64::from(u32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        TY_I32 => non_negative(i64::from(i32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        TY_U64 => Some(u64::from_le_bytes(bytes)),
        TY_I64 => non_negative(i64::from_le_bytes(bytes)),
        _ => None,
    };
    Ok(value)
}

fn non_negative(value: i64) -> Option<u64> {
    u64::try_from(value).ok()
}

/// Consumes one value, keeping it only when `capture` is set and it's an
/// integer or an integer array.
fn consume_value<R: Read + Seek>(
    r: &mut BufReader<R>,
    total_len: u64,
    ty: u32,
    capture: bool,
    depth: u32,
) -> Result<Option<Captured>, GgufError> {
    if is_int(ty) {
        if capture {
            return Ok(read_int(r, ty)?.map(Captured::Int));
        }
        let size = fixed_size(ty).unwrap_or(0);
        skip_bytes(r, total_len, size)?;
        return Ok(None);
    }
    if let Some(size) = fixed_size(ty) {
        skip_bytes(r, total_len, size)?;
        return Ok(None);
    }
    match ty {
        TY_STRING => {
            skip_string(r, total_len)?;
            Ok(None)
        }
        TY_ARRAY => consume_array(r, total_len, capture, depth),
        other => Err(GgufError::Corrupt(format!("unknown value type {other}"))),
    }
}

/// Consumes an array's elements, skipping whatever isn't needed.
fn consume_array<R: Read + Seek>(
    r: &mut BufReader<R>,
    total_len: u64,
    capture: bool,
    depth: u32,
) -> Result<Option<Captured>, GgufError> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::Corrupt(
            "arrays are nested too deeply".to_string(),
        ));
    }
    let elem = read_u32(r)?;
    let count = read_u64(r)?;

    // A cheap lower bound on the array's size stops a huge count before any
    // looping or allocation.
    let min_size = elem_min_size(elem)
        .ok_or_else(|| GgufError::Corrupt(format!("unknown array element type {elem}")))?;
    let minimum = count
        .checked_mul(min_size)
        .ok_or_else(|| GgufError::Corrupt("an array count overflows".to_string()))?;
    let pos = r.stream_position()?;
    checked_end(pos, minimum, total_len)?;

    if let Some(size) = fixed_size(elem) {
        if capture && is_int(elem) && count <= MAX_ARRAY_ELEMENTS {
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                if let Some(value) = read_int(r, elem)? {
                    values.push(value);
                }
            }
            return Ok(Some(Captured::IntArray(values)));
        }
        let bytes = count
            .checked_mul(size)
            .ok_or_else(|| GgufError::Corrupt("an array count overflows".to_string()))?;
        skip_bytes(r, total_len, bytes)?;
        return Ok(None);
    }

    match elem {
        TY_STRING => {
            for _ in 0..count {
                skip_string(r, total_len)?;
            }
            Ok(None)
        }
        TY_ARRAY => {
            for _ in 0..count {
                consume_array(r, total_len, false, depth + 1)?;
            }
            Ok(None)
        }
        other => Err(GgufError::Corrupt(format!(
            "unknown array element type {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Builds GGUF bytes from key/value entries.
    struct Gguf {
        version: u32,
        tensor_count: u64,
        kv_count: u64,
        body: Vec<u8>,
    }

    impl Gguf {
        fn new() -> Self {
            Self {
                version: 3,
                tensor_count: 0,
                kv_count: 0,
                body: Vec::new(),
            }
        }

        fn entry(&mut self, key: &str) -> &mut Self {
            self.kv_count += 1;
            self.push_str(key);
            self
        }

        fn push_str(&mut self, text: &str) {
            self.body
                .extend_from_slice(&(text.len() as u64).to_le_bytes());
            self.body.extend_from_slice(text.as_bytes());
        }

        fn value(&mut self, ty: u32, payload: &[u8]) -> &mut Self {
            self.body.extend_from_slice(&ty.to_le_bytes());
            self.body.extend_from_slice(payload);
            self
        }

        fn u32v(&mut self, value: u32) -> &mut Self {
            self.value(TY_U32, &value.to_le_bytes())
        }

        fn i32v(&mut self, value: i32) -> &mut Self {
            self.value(TY_I32, &value.to_le_bytes())
        }

        fn u64v(&mut self, value: u64) -> &mut Self {
            self.value(TY_U64, &value.to_le_bytes())
        }

        fn string_v(&mut self, text: &str) -> &mut Self {
            let mut payload = (text.len() as u64).to_le_bytes().to_vec();
            payload.extend_from_slice(text.as_bytes());
            self.value(TY_STRING, &payload)
        }

        fn array(&mut self, elem: u32, count: u64) -> &mut Self {
            let mut payload = elem.to_le_bytes().to_vec();
            payload.extend_from_slice(&count.to_le_bytes());
            self.value(TY_ARRAY, &payload)
        }

        fn raw_u32(&mut self, value: u32) -> &mut Self {
            self.body.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn raw_u64(&mut self, value: u64) -> &mut Self {
            self.body.extend_from_slice(&value.to_le_bytes());
            self
        }

        /// Writes a nested array header, as an element of an array of arrays.
        fn nested_array(&mut self, elem: u32, count: u64) -> &mut Self {
            self.raw_u32(elem);
            self.raw_u64(count);
            self
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = MAGIC.to_vec();
            out.extend_from_slice(&self.version.to_le_bytes());
            out.extend_from_slice(&self.tensor_count.to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.body);
            out
        }
    }

    fn parse(bytes: &[u8]) -> Result<ModelMeta, GgufError> {
        read_gguf_meta_from(Cursor::new(bytes.to_vec()), bytes.len() as u64)
    }

    fn llama_file() -> Gguf {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("llama");
        g.entry("llama.context_length").u32v(8192);
        g.entry("llama.block_count").u32v(32);
        g.entry("llama.embedding_length").u32v(4096);
        g.entry("llama.attention.head_count").u32v(32);
        g.entry("llama.attention.head_count_kv").u32v(8);
        g.entry("llama.attention.key_length").u32v(128);
        g.entry("llama.attention.value_length").u32v(128);
        g.entry("llama.attention.sliding_window").u32v(4096);
        g
    }

    #[test]
    fn reads_every_wanted_key() {
        let meta = parse(&llama_file().bytes()).expect("a normal file parses");

        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, Some(8192));
        assert_eq!(meta.block_count, Some(32));
        assert_eq!(meta.embedding_length, Some(4096));
        assert_eq!(meta.head_count, Some(32));
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.key_length, Some(128));
        assert_eq!(meta.value_length, Some(128));
        assert_eq!(meta.sliding_window, Some(4096));

        assert_eq!(meta.kv_heads(), Some(8));
        assert_eq!(meta.key_dim(), Some(128));
        assert_eq!(meta.value_dim(), Some(128));
    }

    #[test]
    fn fallbacks_apply_without_kv_heads_and_key_length() {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("qwen2");
        g.entry("qwen2.embedding_length").u32v(4096);
        g.entry("qwen2.attention.head_count").u32v(32);

        let meta = parse(&g.bytes()).expect("a partial file parses");
        assert_eq!(meta.head_count_kv, None);
        assert_eq!(meta.key_length, None);
        assert_eq!(meta.kv_heads(), Some(32));
        assert_eq!(meta.key_dim(), Some(128));
        assert_eq!(meta.value_dim(), Some(128));
    }

    #[test]
    fn a_per_layer_head_count_uses_its_largest_element() {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("llama");
        g.entry("llama.attention.head_count_kv").array(TY_U32, 4);
        g.raw_u32(8).raw_u32(4).raw_u32(2).raw_u32(8);

        let meta = parse(&g.bytes()).expect("an array-valued key parses");
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.kv_heads(), Some(8));
    }

    #[test]
    fn skips_string_and_nested_arrays_before_the_wanted_keys() {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("llama");
        g.entry("llama.some.strings").array(TY_STRING, 2);
        g.raw_u64(3);
        g.body.extend_from_slice(b"abc");
        g.raw_u64(0);
        g.entry("llama.some.matrix").array(TY_ARRAY, 2);
        g.nested_array(TY_U32, 2).raw_u32(1).raw_u32(2);
        g.nested_array(TY_U32, 1).raw_u32(3);
        g.entry("llama.attention.head_count").u32v(32);

        let meta = parse(&g.bytes()).expect("skipped values don't stop the parse");
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.head_count, Some(32));
    }

    #[test]
    fn keys_before_the_architecture_are_resolved() {
        let mut g = Gguf::new();
        g.entry("llama.context_length").u32v(4096);
        g.entry("llama.block_count").u32v(16);
        g.entry("general.architecture").string_v("llama");

        let meta = parse(&g.bytes()).expect("late architecture keys resolve");
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, Some(4096));
        assert_eq!(meta.block_count, Some(16));
    }

    #[test]
    fn stops_at_a_tokenizer_key_once_the_architecture_is_known() {
        let mut g = llama_file();
        g.entry("tokenizer.ggml.tokens")
            .array(TY_STRING, 4_000_000_000);

        let meta = parse(&g.bytes()).expect("the huge tokenizer array is never read");
        assert_eq!(meta.context_length, Some(8192));
        assert_eq!(meta.block_count, Some(32));
    }

    #[test]
    fn negative_numbers_leave_fields_unknown() {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("llama");
        g.entry("llama.context_length").i32v(-1);
        g.entry("llama.block_count").u64v(24);

        let meta = parse(&g.bytes()).expect("negative values parse");
        assert_eq!(meta.context_length, None);
        assert_eq!(meta.block_count, Some(24));
    }

    #[test]
    fn a_string_past_the_end_is_corrupt() {
        let mut g = Gguf::new();
        g.entry("general.architecture").string_v("llama");
        g.entry("llama.vocabulary")
            .value(TY_STRING, &1_000_000_000u64.to_le_bytes());

        let err = parse(&g.bytes()).expect_err("an oversized string is refused");
        assert!(matches!(err, GgufError::Corrupt(_)), "{err:?}");
    }

    #[test]
    fn a_huge_key_length_is_corrupt() {
        let mut g = Gguf::new();
        g.kv_count += 1;
        g.raw_u64(1_000_000_000);

        let err = parse(&g.bytes()).expect_err("an oversized key is refused");
        assert!(matches!(err, GgufError::Corrupt(_)), "{err:?}");
    }

    #[test]
    fn a_truncated_stream_is_an_error() {
        let bytes = llama_file().bytes();
        let cut = &bytes[..bytes.len() - 3];

        let err = parse(cut).expect_err("a truncated file is refused");
        assert!(
            matches!(err, GgufError::Corrupt(_) | GgufError::Io(_)),
            "{err:?}"
        );
    }

    #[test]
    fn bad_magic_is_not_gguf() {
        let mut bytes = llama_file().bytes();
        bytes[..4].copy_from_slice(b"GGUX");

        let err = parse(&bytes).expect_err("a wrong magic is refused");
        assert!(matches!(err, GgufError::NotGguf), "{err:?}");
    }

    #[test]
    fn version_one_is_unsupported() {
        let mut g = llama_file();
        g.version = 1;

        let err = parse(&g.bytes()).expect_err("version 1 is refused");
        assert!(matches!(err, GgufError::UnsupportedVersion(1)), "{err:?}");
    }

    #[test]
    fn reads_from_a_file_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, llama_file().bytes()).expect("write fixture");

        let meta = read_gguf_meta(&path).expect("a file on disk parses");
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.block_count, Some(32));
        assert_eq!(meta.kv_heads(), Some(8));
    }
}
