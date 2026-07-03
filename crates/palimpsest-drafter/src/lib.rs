//! The Drafter: a minimal autoregressive transformer that writes the
//! initial scratchpad draft (Phase 1).
//!
//! Architecture novelty is explicitly not the point — this exists to
//! validate the training pipeline and to serve as the substrate whose
//! internal activations the Router reads (Phase 4), hence
//! [`Drafter::forward_with_hidden`] is part of the public interface from
//! the start.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use palimpsest_core::nn::{EncoderBlock, EncoderBlockConfig, SeqEmbedding, SeqEmbeddingConfig};
use palimpsest_core::tokenizer::TokenId;

#[derive(Config, Debug)]
pub struct DrafterConfig {
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

impl DrafterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Drafter<B> {
        Drafter {
            embed: SeqEmbeddingConfig::new(self.vocab_size, self.max_seq_len, self.d_model)
                .init(device),
            blocks: (0..self.n_layers)
                .map(|_| {
                    EncoderBlockConfig::new(self.d_model, self.n_heads, self.d_ff).init(device)
                })
                .collect(),
            norm: LayerNormConfig::new(self.d_model).init(device),
            head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Drafter<B: Backend> {
    embed: SeqEmbedding<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> Drafter<B> {
    /// tokens `[batch, seq]` → logits `[batch, seq, vocab]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with_hidden(tokens).0
    }

    /// Like `forward`, but also returns each block's output hidden state
    /// (`[batch, seq, d_model]` per layer, pre-final-norm). This is the
    /// activation hook the Router trains on — kept as a first-class
    /// interface so Phase 6 can call it repeatedly without a forward-pass
    /// rewrite. Tensor clones are refcounted handles, so the overhead of
    /// always collecting them is negligible at toy scale.
    pub fn forward_with_hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let [b, t] = tokens.dims();
        let device = tokens.device();
        let mask = generate_autoregressive_mask::<B>(b, t, &device);
        let mut x = self.embed.forward(tokens);
        let mut hidden = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            x = block.forward(x, Some(mask.clone()));
            hidden.push(x.clone());
        }
        let x = self.norm.forward(x);
        (self.head.forward(x), hidden)
    }

    /// Greedy autoregressive continuation of `prefix` by `num_new` tokens.
    /// Re-runs the full forward each step — fine at toy scale, no KV cache.
    pub fn generate_greedy(
        &self,
        prefix: &[TokenId],
        num_new: usize,
        device: &B::Device,
    ) -> Vec<TokenId> {
        let mut tokens = prefix.to_vec();
        for _ in 0..num_new {
            let t = tokens.len();
            let data: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
            let input = Tensor::<B, 2, Int>::from_data(TensorData::new(data, [1, t]), device);
            let logits = self.forward(input);
            let next: i64 = logits
                .slice([0..1, t - 1..t])
                .argmax(2)
                .into_scalar()
                .elem();
            tokens.push(next as TokenId);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn forward_shapes_and_hidden_states() {
        let device = Default::default();
        let config = DrafterConfig::new(22).with_max_seq_len(16);
        let model = config.init::<B>(&device);
        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![1i64; 2 * 14], [2, 14]),
            &device,
        );
        let (logits, hidden) = model.forward_with_hidden(tokens);
        assert_eq!(logits.dims(), [2, 14, 22]);
        assert_eq!(hidden.len(), 2);
        assert_eq!(hidden[0].dims(), [2, 14, 64]);
    }

    #[test]
    fn greedy_generation_appends_tokens() {
        let device = Default::default();
        let model = DrafterConfig::new(22).with_max_seq_len(16).init::<B>(&device);
        let out = model.generate_greedy(&[1, 6, 14], 3, &device);
        assert_eq!(out.len(), 6);
        assert_eq!(&out[..3], &[1, 6, 14]);
    }
}
