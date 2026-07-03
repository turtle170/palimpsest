//! The Critic: predicts per-token causal "weight" (how much ablating this
//! token would change the output), trained by regression against the
//! Phase 2 ablation labels. Also owns the halting decision for the edit
//! loop.
//!
//! A shallow *bidirectional* transformer over tokens: unlike the Drafter it
//! may look at the whole sequence, since importance of a token depends on
//! what comes after it (a key is important because it is queried later).

use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use palimpsest_core::nn::{EncoderBlock, EncoderBlockConfig, SeqEmbedding, SeqEmbeddingConfig};
use palimpsest_core::span::Span;

#[derive(Config, Debug)]
pub struct CriticConfig {
    pub vocab_size: usize,
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 64)]
    pub max_seq_len: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Critic<B> {
        Critic {
            embed: SeqEmbeddingConfig::new(self.vocab_size, self.max_seq_len, self.d_model)
                .init(device),
            blocks: (0..self.n_layers)
                .map(|_| {
                    EncoderBlockConfig::new(self.d_model, self.n_heads, self.d_ff).init(device)
                })
                .collect(),
            norm: LayerNormConfig::new(self.d_model).init(device),
            head: LinearConfig::new(self.d_model, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    embed: SeqEmbedding<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    /// tokens `[batch, seq]` → predicted per-token weight `[batch, seq]`.
    ///
    /// Output is unbounded; labels are non-negative NLL deltas, so a
    /// well-trained Critic emits ~0 for filler. No output activation on
    /// purpose — clamping happens at the consumer if needed.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let mut x = self.embed.forward(tokens);
        for block in &self.blocks {
            x = block.forward(x, None); // bidirectional
        }
        let x = self.norm.forward(x);
        self.head.forward(x).squeeze_dims(&[2])
    }
}

/// Halting rule for the edit loop (Phase 3 requirement, consumed by
/// Phase 6): stop editing once the largest remaining flagged-span weight
/// falls below the threshold — i.e. nothing left on the pad is worth
/// touching.
#[derive(Config, Debug)]
pub struct HaltingConfig {
    /// Tunable; default chosen empirically against toy-task label scale
    /// (queried-pair deltas are ~1-4 nats, filler ~0).
    #[config(default = 0.05)]
    pub weight_threshold: f64,
}

impl HaltingConfig {
    /// `true` = stop editing. Halts when no spans remain or all remaining
    /// spans are below threshold.
    pub fn should_halt(&self, remaining_span_weights: &[f32]) -> bool {
        remaining_span_weights
            .iter()
            .all(|&w| (w as f64) < self.weight_threshold)
    }
}

/// Mean predicted weight over a span — the scalar the loop compares
/// against the halting threshold and uses to rank spans.
pub fn span_weight(token_weights: &[f32], span: Span) -> f32 {
    let slice = &token_weights[span.start..span.end];
    slice.iter().sum::<f32>() / slice.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn forward_shape() {
        let device = Default::default();
        let critic = CriticConfig::new(22).with_max_seq_len(16).init::<B>(&device);
        let tokens =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i64; 28], [2, 14]), &device);
        assert_eq!(critic.forward(tokens).dims(), [2, 14]);
    }

    #[test]
    fn halting() {
        let cfg = HaltingConfig::new();
        assert!(cfg.should_halt(&[]));
        assert!(cfg.should_halt(&[0.01, 0.002]));
        assert!(!cfg.should_halt(&[0.01, 0.8]));
    }

    #[test]
    fn span_weight_means_over_span() {
        let w = span_weight(&[0.0, 1.0, 3.0, 0.0], Span::new(1, 3));
        assert!((w - 2.0).abs() < 1e-6);
    }
}
