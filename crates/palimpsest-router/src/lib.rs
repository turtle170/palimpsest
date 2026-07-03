//! The Router: reads the Drafter's internal hidden states and emits a
//! per-token "attention-worthy" score — where should the Editor look
//! (Phase 4).
//!
//! Deliberately a per-position MLP with no attention of its own: the whole
//! point is to be *cheap* relative to a full Critic/ablation pass, and the
//! Drafter's hidden states already carry contextual information. If
//! per-position readout proves too weak, widening to a small attention
//! layer is the first thing to try.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Config, Debug)]
pub struct RouterConfig {
    /// Width of the Drafter hidden states this Router reads (d_model).
    pub d_input: usize,
    #[config(default = 64)]
    pub d_hidden: usize,
}

impl RouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Router<B> {
        Router {
            l1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            l2: LinearConfig::new(self.d_hidden, 1).init(device),
            act: Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    act: Gelu,
}

impl<B: Backend> Router<B> {
    /// Drafter hidden states `[batch, seq, d_input]` → scores `[batch, seq]`.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = self.act.forward(self.l1.forward(hidden));
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
