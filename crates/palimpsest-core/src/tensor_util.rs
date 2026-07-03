//! Tensor helpers shared across component crates.
//!
//! The central one is [`sequence_nll`]: per-position negative log-likelihood.
//! Training losses take its mean; the ablation pipeline (Phase 2) diffs it
//! position-by-position between clean and ablated inputs, so it must stay
//! un-reduced here.

use burn::tensor::activation::log_softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Per-position NLL of `targets` under `logits`.
///
/// Shapes: logits `[batch, seq, vocab]`, targets `[batch, seq]` (Int) →
/// output `[batch, seq]` (f32, >= 0).
pub fn sequence_nll<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let log_probs = log_softmax(logits, 2);
    let gathered = log_probs.gather(2, targets.unsqueeze_dim(2));
    let nll: Tensor<B, 2> = gathered.squeeze_dims(&[2]);
    nll.neg()
}

/// Mean NLL over all positions — the standard next-token training loss.
pub fn mean_nll<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
    sequence_nll(logits, targets).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn nll_prefers_correct_target() {
        let device = Default::default();
        // Two positions; logits strongly favor class 1 then class 0.
        let logits = Tensor::<B, 3>::from_data(
            TensorData::from([[[0.0f32, 5.0, 0.0], [5.0, 0.0, 0.0]]]),
            &device,
        );
        let good = Tensor::<B, 2, Int>::from_data(TensorData::from([[1i64, 0]]), &device);
        let bad = Tensor::<B, 2, Int>::from_data(TensorData::from([[2i64, 1]]), &device);

        let nll_good = sequence_nll(logits.clone(), good).mean().into_scalar();
        let nll_bad = sequence_nll(logits, bad).mean().into_scalar();
        assert!(nll_good < 0.1, "nll_good={nll_good}");
        assert!(nll_bad > 2.0, "nll_bad={nll_bad}");
    }
}
