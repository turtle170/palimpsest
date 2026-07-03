use std::collections::HashMap;

/// Token ids are plain u32 everywhere outside tensors; conversion to the
/// backend's int type happens only at batch-construction time.
pub type TokenId = u32;

/// Minimal tokenizer interface, established early (Phase 0) so that a later
/// swap to a real BPE tokenizer does not ripple through every crate.
///
/// Special-token accessors return `Option` because not every tokenizer needs
/// every special token; the ablation pipeline requires `mask_id`.
pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn pad_id(&self) -> TokenId;
    fn bos_id(&self) -> Option<TokenId>;
    fn eos_id(&self) -> Option<TokenId>;
    /// Token used to ablate/mask spans (Phase 2) and as Editor input
    /// corruption (Phase 5).
    fn mask_id(&self) -> Option<TokenId>;
}

/// Trivial char-level tokenizer over a fixed alphabet. First concrete impl
/// of the trait; the KV toy task uses its own symbolic vocab (see
/// palimpsest-data), and real BPE comes much later.
pub struct CharTokenizer {
    chars: Vec<char>,
    ids: HashMap<char, TokenId>,
}

/// Layout: 0=PAD, 1=BOS, 2=EOS, 3=MASK, then alphabet chars in order.
const NUM_SPECIAL: TokenId = 4;

impl CharTokenizer {
    pub fn new(alphabet: &str) -> Self {
        let chars: Vec<char> = alphabet.chars().collect();
        let ids = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as TokenId + NUM_SPECIAL))
            .collect();
        Self { chars, ids }
    }
}

impl Tokenizer for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.chars.len() + NUM_SPECIAL as usize
    }

    fn encode(&self, text: &str) -> Vec<TokenId> {
        // Unknown characters are silently dropped — acceptable for a toy
        // tokenizer over a controlled alphabet.
        text.chars().filter_map(|c| self.ids.get(&c).copied()).collect()
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .filter_map(|&t| {
                t.checked_sub(NUM_SPECIAL)
                    .and_then(|i| self.chars.get(i as usize))
            })
            .collect()
    }

    fn pad_id(&self) -> TokenId {
        0
    }
    fn bos_id(&self) -> Option<TokenId> {
        Some(1)
    }
    fn eos_id(&self) -> Option<TokenId> {
        Some(2)
    }
    fn mask_id(&self) -> Option<TokenId> {
        Some(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_roundtrip() {
        let tok = CharTokenizer::new("abc");
        let ids = tok.encode("abcba");
        assert_eq!(ids.len(), 5);
        assert_eq!(tok.decode(&ids), "abcba");
        assert_eq!(tok.vocab_size(), 7);
    }
}
