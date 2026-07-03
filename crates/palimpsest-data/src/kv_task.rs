use palimpsest_core::span::Span;
use palimpsest_core::tokenizer::TokenId;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::vocab::{self, KvVocab};

/// Configuration for the key-value recall toy task.
///
/// Sequence layout (fixed length = 6 + 2 * num_pairs):
/// `BOS  k1 v1  k2 v2  ...  SEP  kq QUERY  answer  EOS`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvTaskConfig {
    pub num_keys: usize,
    pub num_values: usize,
    /// Pairs per sequence; keys within a sequence are distinct so every
    /// query has a unique correct answer.
    pub num_pairs: usize,
}

impl Default for KvTaskConfig {
    fn default() -> Self {
        Self {
            num_keys: 8,
            num_values: 8,
            num_pairs: 4,
        }
    }
}

impl KvTaskConfig {
    pub fn vocab(&self) -> KvVocab {
        KvVocab::new(self.num_keys, self.num_values)
    }

    pub fn seq_len(&self) -> usize {
        6 + 2 * self.num_pairs
    }

    /// Index of the answer token (the value the model must recall).
    pub fn answer_pos(&self) -> usize {
        self.seq_len() - 2
    }
}

/// One generated example plus the metadata needed for sanity checks:
/// which pair was queried and where each pair sits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvExample {
    pub tokens: Vec<TokenId>,
    pub answer_pos: usize,
    /// Span of each (key, value) pair, in order of appearance.
    pub pair_spans: Vec<Span>,
    /// Index into `pair_spans` of the queried pair.
    pub queried_pair: usize,
}

impl KvExample {
    /// The span of the pair that causally determines the answer.
    pub fn queried_span(&self) -> Span {
        self.pair_spans[self.queried_pair]
    }
}

/// Deterministically generate `n` examples from `seed`. Train/valid splits
/// use different seeds; with ~27M distinct sequences at default config,
/// split contamination is statistically negligible at toy scale.
pub fn generate_examples(config: &KvTaskConfig, n: usize, seed: u64) -> Vec<KvExample> {
    assert!(
        config.num_pairs <= config.num_keys,
        "need distinct keys per sequence"
    );
    let vocab = config.vocab();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_keys: Vec<usize> = (0..config.num_keys).collect();

    (0..n)
        .map(|_| {
            all_keys.shuffle(&mut rng);
            let keys = &all_keys[..config.num_pairs];
            let values: Vec<usize> = (0..config.num_pairs)
                .map(|_| rng.random_range(0..config.num_values))
                .collect();
            let queried_pair = rng.random_range(0..config.num_pairs);

            let mut tokens = Vec::with_capacity(config.seq_len());
            let mut pair_spans = Vec::with_capacity(config.num_pairs);
            tokens.push(vocab::BOS);
            for (&k, &v) in keys.iter().zip(&values) {
                let start = tokens.len();
                tokens.push(vocab.key_id(k));
                tokens.push(vocab.value_id(v));
                pair_spans.push(Span::new(start, start + 2));
            }
            tokens.push(vocab::SEP);
            tokens.push(vocab.key_id(keys[queried_pair]));
            tokens.push(vocab::QUERY);
            tokens.push(vocab.value_id(values[queried_pair]));
            tokens.push(vocab::EOS);
            debug_assert_eq!(tokens.len(), config.seq_len());

            KvExample {
                tokens,
                answer_pos: config.answer_pos(),
                pair_spans,
                queried_pair,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use palimpsest_core::tokenizer::Tokenizer;

    #[test]
    fn generates_consistent_examples() {
        let config = KvTaskConfig::default();
        let examples = generate_examples(&config, 100, 42);

        for ex in &examples {
            assert_eq!(ex.tokens.len(), config.seq_len());
            assert_eq!(ex.tokens[0], vocab::BOS);
            assert_eq!(*ex.tokens.last().unwrap(), vocab::EOS);
            // The answer must equal the value of the queried pair.
            let qspan = ex.queried_span();
            assert_eq!(ex.tokens[ex.answer_pos], ex.tokens[qspan.start + 1]);
            // The query key must equal the queried pair's key.
            assert_eq!(ex.tokens[ex.answer_pos - 2], ex.tokens[qspan.start]);
            // Keys distinct within the sequence.
            let keys: Vec<TokenId> = ex.pair_spans.iter().map(|s| ex.tokens[s.start]).collect();
            let mut dedup = keys.clone();
            dedup.dedup();
            assert_eq!(keys.len(), {
                dedup.sort_unstable();
                dedup.dedup();
                dedup.len()
            });
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let config = KvTaskConfig::default();
        let a = generate_examples(&config, 10, 7);
        let b = generate_examples(&config, 10, 7);
        assert_eq!(
            a.iter().map(|e| e.tokens.clone()).collect::<Vec<_>>(),
            b.iter().map(|e| e.tokens.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn decodes_to_readable_form() {
        let config = KvTaskConfig::default();
        let vocab = config.vocab();
        let ex = &generate_examples(&config, 1, 0)[0];
        let text = vocab.decode(&ex.tokens);
        assert!(text.starts_with("BOS"), "{text}");
        assert!(text.contains("QUERY"), "{text}");
    }
}
