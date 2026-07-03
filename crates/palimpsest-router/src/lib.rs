//! The Router: reads the Drafter's internal hidden states and emits a
//! per-token "attention-worthy" score — where should the Editor look
//! (Phase 4).
//!
//! Design history (empirical, on the KV toy task): the first version was a
//! pure per-position MLP, which failed its checkpoint (top-3 overlap with
//! ablation truth 0.41 vs 0.26 shuffled baseline). Root cause: the Drafter
//! is *causal*, so its activation at position p cannot encode importance
//! that depends on later tokens (a pair matters because it is queried
//! afterwards). The Router therefore needs cross-position mixing of the
//! activations; one small bidirectional attention block suffices and keeps
//! it much cheaper than a full Critic pass (which must re-encode raw
//! tokens from scratch — the Router amortizes the Drafter's compute).

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use palimpsest_core::nn::{EncoderBlock, EncoderBlockConfig};

#[derive(Config, Debug)]
pub struct RouterConfig {
    /// Width of the Drafter hidden states this Router reads (d_model).
    pub d_input: usize,
    #[config(default = 64)]
    pub d_hidden: usize,
    #[config(default = 4)]
    pub n_heads: usize,
}

impl RouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Router<B> {
        Router {
            proj: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            block: EncoderBlockConfig::new(self.d_hidden, self.n_heads, self.d_hidden * 4)
                .init(device),
            norm: LayerNormConfig::new(self.d_hidden).init(device),
            l1: LinearConfig::new(self.d_hidden, self.d_hidden).init(device),
            l2: LinearConfig::new(self.d_hidden, 1).init(device),
            act: Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    proj: Linear<B>,
    /// Bidirectional: routes future-position information backwards, which
    /// causal Drafter activations cannot carry themselves.
    block: EncoderBlock<B>,
    norm: LayerNorm<B>,
    l1: Linear<B>,
    l2: Linear<B>,
    act: Gelu,
}

impl<B: Backend> Router<B> {
    /// Drafter hidden states `[batch, seq, d_input]` → scores `[batch, seq]`.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = self.proj.forward(hidden);
        let h = self.block.forward(h, None);
        let h = self.norm.forward(h);
        let h = self.act.forward(self.l1.forward(h));
        self.l2.forward(h).squeeze_dims(&[2])
    }
}

/// Positions whose score exceeds `threshold` — the flagging rule the edit
/// loop feeds into span merging.
pub fn flag_positions(scores: &[f32], threshold: f32) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s > threshold)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn forward_shape() {
        let device = Default::default();
        let router = RouterConfig::new(64).init::<B>(&device);
        let hidden = Tensor::<B, 3>::zeros([2, 14, 64], &device);
        assert_eq!(router.forward(hidden).dims(), [2, 14]);
    }

    #[test]
    fn flagging() {
        assert_eq!(flag_positions(&[0.1, 0.9, 0.05, 0.7], 0.5), vec![1, 3]);
    }
}
