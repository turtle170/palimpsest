pub mod ablate;
pub mod critic;
pub mod drafter;
pub mod editor;
pub mod router;
pub mod run_loop;

use std::io::BufRead;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use palimpsest_data::{KvExample, LabeledSequence, TokenBatch};

/// Load a Phase 2 label file (one JSON LabeledSequence per line).
pub fn load_labels(path: &Path) -> anyhow::Result<Vec<LabeledSequence>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open {}: {e} (run `ablate` first?)", path.display()))?;
    std::io::BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

/// Build one batch tensor over the given example indices.
pub fn batch_tokens<B: Backend>(
    examples: &[KvExample],
    indices: &[usize],
    device: &B::Device,
) -> TokenBatch<B> {
    let seqs: Vec<&[u32]> = indices
        .iter()
        .map(|&i| examples[i].tokens.as_slice())
        .collect();
    TokenBatch::from_sequences(&seqs, device)
}

/// Fraction of examples where the greedy prediction at the answer position
/// matches the true bound value — the task-level quality metric used by
/// several phases.
pub fn answer_accuracy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    answer_pos: usize,
) -> f32 {
    let [b, _t, _v] = logits.dims();
    // Prediction index answer_pos - 1 predicts the token at answer_pos.
    let idx = answer_pos - 1;
    let preds = logits
        .slice([0..b, idx..idx + 1])
        .argmax(2)
        .reshape([b as i32]);
    let truth = targets.slice([0..b, idx..idx + 1]).reshape([b as i32]);
    let correct: f32 = preds.equal(truth).int().sum().into_scalar().elem();
    correct / b as f32
}
