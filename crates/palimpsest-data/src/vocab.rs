use palimpsest_core::tokenizer::{TokenId, Tokenizer};
use serde::{Deserialize, Serialize};

pub const PAD: TokenId = 0;
pub const BOS: TokenId = 1;
pub const EOS: TokenId = 2;
pub const SEP: TokenId = 3;
pub const QUERY: TokenId = 4;
pub const MASK: TokenId = 5;
pub const NUM_SPECIAL: TokenId = 6;

/// Symbolic vocabulary for the key-value recall task:
/// `[PAD, BOS, EOS, SEP, QUERY, MASK, K0..K{num_keys-1}, V0..V{num_values-1}]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvVocab {
    pub num_keys: usize,
    pub num_values: usize,
}

impl KvVocab {
    pub fn new(num_keys: usize, num_values: usize) -> Self {
        Self { num_keys, num_values }
    }

    pub fn key_id(&self, k: usize) -> TokenId {
        assert!(k < self.num_keys);
        NUM_SPECIAL + k as TokenId
    }

    pub fn value_id(&self, v: usize) -> TokenId {
        assert!(v < self.num_values);
        NUM_SPECIAL + self.num_keys as TokenId + v as TokenId
    }

    pub fn is_key(&self, id: TokenId) -> bool {
        (NUM_SPECIAL..NUM_SPECIAL + self.num_keys as TokenId).contains(&id)
    }

    pub fn is_value(&self, id: TokenId) -> bool {
        let lo = NUM_SPECIAL + self.num_keys as TokenId;
        (lo..lo + self.num_values as TokenId).contains(&id)
    }

    pub fn token_name(&self, id: TokenId) -> String {
        match id {
            PAD => "PAD".into(),
            BOS => "BOS".into(),
            EOS => "EOS".into(),
            SEP => "SEP".into(),
            QUERY => "QUERY".into(),
            MASK => "MASK".into(),
            _ if self.is_key(id) => format!("K{}", id - NUM_SPECIAL),
            _ if self.is_value(id) => format!("V{}", id - NUM_SPECIAL - self.num_keys as TokenId),
            _ => format!("<invalid:{id}>"),
        }
    }
}

impl Tokenizer for KvVocab {
    fn vocab_size(&self) -> usize {
        NUM_SPECIAL as usize + self.num_keys + self.num_values
    }

    /// Encodes whitespace-separated symbolic token names ("BOS K3 V1 ...").
    fn encode(&self, text: &str) -> Vec<TokenId> {
        text.split_whitespace()
            .filter_map(|name| self.id_of_name(name))
            .collect()
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|&t| self.token_name(t))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn pad_id(&self) -> TokenId {
        PAD
    }
    fn bos_id(&self) -> Option<TokenId> {
        Some(BOS)
    }
    fn eos_id(&self) -> Option<TokenId> {
        Some(EOS)
    }
    fn mask_id(&self) -> Option<TokenId> {
        Some(MASK)
    }
}

impl KvVocab {
    fn id_of_name(&self, name: &str) -> Option<TokenId> {
        match name {
            "PAD" => Some(PAD),
            "BOS" => Some(BOS),
            "EOS" => Some(EOS),
            "SEP" => Some(SEP),
            "QUERY" => Some(QUERY),
            "MASK" => Some(MASK),
            _ => {
                let (prefix, idx) = name.split_at(1);
                let idx: usize = idx.parse().ok()?;
                match prefix {
                    "K" if idx < self.num_keys => Some(self.key_id(idx)),
                    "V" if idx < self.num_values => Some(self.value_id(idx)),
                    _ => None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let vocab = KvVocab::new(8, 8);
        let text = "BOS K3 V1 K0 V7 SEP K3 QUERY V1 EOS";
        let ids = vocab.encode(text);
        assert_eq!(vocab.decode(&ids), text);
        assert_eq!(vocab.vocab_size(), 22);
    }
}
