//! Lightweight batching: turn token sequences into Burn Int tensors.
//!
//! We intentionally skip Burn's Dataset/DataLoader machinery — the toy data
//! is tiny, fixed-length, and generated in memory, so a plain shuffled
//! index walk is simpler and keeps the data crate backend-agnostic.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use palimpsest_core::tokenizer::TokenId;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// A batch of equal-length token sequences as an Int tensor `[batch, seq]`.
pub struct TokenBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
}

impl<B: Backend> TokenBatch<B> {
    pub fn from_sequences(seqs: &[&[TokenId]], device: &B::Device) -> Self {
        assert!(!seqs.is_empty());
        let seq_len = seqs[0].len();
        assert!(
            seqs.iter().all(|s| s.len() == seq_len),
            "all sequences in a batch must have equal length"
        );
        let flat: Vec<i64> = seqs
            .iter()
            .flat_map(|s| s.iter().map(|&t| t as i64))
            .collect();
        let tokens = Tensor::from_data(
            TensorData::new(flat, [seqs.len(), seq_len]),
            device,
        );
        Self { tokens }
    }

    /// Next-token training views: inputs `[batch, seq-1]` and targets
    /// `[batch, seq-1]` (targets are inputs shifted left by one).
    pub fn autoregressive_views(&self) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let [b, t] = self.tokens.dims();
        let inputs = self.tokens.clone().slice([0..b, 0..t - 1]);
        let targets = self.tokens.clone().slice([0..b, 1..t]);
        (inputs, targets)
    }
}

/// Yields shuffled minibatches of indices, reshuffling each epoch.
pub struct BatchIndexer {
    indices: Vec<usize>,
    batch_size: usize,
    cursor: usize,
    rng: StdRng,
}

impl BatchIndexer {
    pub fn new(n: usize, batch_size: usize, seed: u64) -> Self {
        assert!(batch_size > 0 && n > 0);
        let mut s = Self {
            indices: (0..n).collect(),
            batch_size,
            cursor: 0,
            rng: StdRng::seed_from_u64(seed),
        };
        s.indices.shuffle(&mut s.rng);
        s
    }

    /// Next minibatch of indices; reshuffles and wraps at epoch end
    /// (drops the ragged tail batch).
    pub fn next_batch(&mut self) -> &[usize] {
        if self.cursor + self.batch_size > self.indices.len() {
            self.indices.shuffle(&mut self.rng);
            self.cursor = 0;
        }
        let batch = &self.indices[self.cursor..self.cursor + self.batch_size];
        self.cursor += self.batch_size;
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_task::{KvTaskConfig, generate_examples};
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn builds_batch_with_expected_shapes() {
        let config = KvTaskConfig::default();
        let examples = generate_examples(&config, 8, 0);
        let device = Default::default();
        let seqs: Vec<&[u32]> = examples.iter().map(|e| e.tokens.as_slice()).collect();
        let batch = TokenBatch::<B>::from_sequences(&seqs, &device);
        assert_eq!(batch.tokens.dims(), [8, config.seq_len()]);

        let (inputs, targets) = batch.autoregressive_views();
        assert_eq!(inputs.dims(), [8, config.seq_len() - 1]);
        assert_eq!(targets.dims(), [8, config.seq_len() - 1]);

        // Target row 0 must equal input row 0 shifted by one.
        let inp: Vec<i64> = inputs.slice([0..1, 0..4]).into_data().to_vec().unwrap();
        let tgt: Vec<i64> = targets.slice([0..1, 0..3]).into_data().to_vec().unwrap();
        assert_eq!(inp[1..4], tgt[..]);
    }

    #[test]
    fn indexer_cycles_without_repeats_within_epoch() {
        let mut idx = BatchIndexer::new(10, 3, 1);
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.extend_from_slice(idx.next_batch());
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 9, "no repeats within one epoch");
    }
}
