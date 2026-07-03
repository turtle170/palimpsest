//! The Editor: given a sequence and a flagged span, emit a replacement
//! patch (Phase 5).
//!
//! v1 design decisions (all explicitly provisional — this is the least
//! well-specified component in the plan and the first working version is a
//! baseline to iterate on, not a final design):
//!
//! * **Same-length patches only.** The Editor predicts one replacement
//!   token per span position, masked-LM style, rather than decoding a
//!   variable-length patch autoregressively. Keeps indices stable across
//!   edits and the training objective trivial. OPEN QUESTION: variable
//!   length patches, and what the training objective should be once we
//!   move past corrupt-and-repair (e.g. improving weak-but-valid drafts
//!   rather than repairing corruptions).
//! * **Full-sequence context.** The Editor sees the whole (short, toy)
//!   sequence with an in-span indicator embedding, not a clipped window.
//!   OPEN QUESTION: local windows once sequences outgrow toy scale.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use palimpsest_core::nn::{EncoderBlock, EncoderBlockConfig, SeqEmbedding, SeqEmbeddingConfig};
use palimpsest_core::patch::Patch;
use palimpsest_core::span::Span;
use palimpsest_core::tokenizer::TokenId;

#[derive(Config, Debug)]
pub struct EditorConfig {
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

impl EditorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Editor<B> {
        Editor {
            embed: SeqEmbeddingConfig::new(self.vocab_size, self.max_seq_len, self.d_model)
                .init(device),
            // Two entries: 0 = context position, 1 = inside flagged span.
            span_flag: EmbeddingConfig::new(2, self.d_model).init(device),
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
pub struct Editor<B: Backend> {
    embed: SeqEmbedding<B>,
    span_flag: Embedding<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> Editor<B> {
    /// tokens `[batch, seq]`, span_mask `[batch, seq]` (Int, 1 = in-span)
    /// → per-position replacement logits `[batch, seq, vocab]`.
    /// Loss is only taken on in-span positions; out-of-span logits are
    /// unused.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, span_mask: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embed.forward(tokens) + self.span_flag.forward(span_mask);
        for block in &self.blocks {
            x = block.forward(x, None); // bidirectional
        }
        let x = self.norm.forward(x);
        self.head.forward(x)
    }

    /// Propose a same-length replacement patch for `span` (greedy argmax
    /// per position).
    pub fn propose_patch(
        &self,
        tokens: &[TokenId],
        span: Span,
        device: &B::Device,
    ) -> Patch {
        let t = tokens.len();
        assert!(span.end <= t && !span.is_empty());
        let data: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(data, [1, t]), device);
        let mask_data: Vec<i64> = (0..t).map(|i| span.contains(i) as i64).collect();
        let span_mask =
            Tensor::<B, 2, Int>::from_data(TensorData::new(mask_data, [1, t]), device);

        let logits = self.forward(input, span_mask); // [1, t, vocab]
        let preds = logits.argmax(2); // [1, t, 1]
        let pred_vec: Vec<i64> = preds.reshape([t]).into_data().to_vec().unwrap();
        let replacement = span.positions().map(|p| pred_vec[p] as TokenId).collect();
        Patch::new(span, replacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn proposes_same_length_patch() {
        let device = Default::default();
        let editor = EditorConfig::new(22).with_max_seq_len(16).init::<B>(&device);
        let tokens: Vec<TokenId> = vec![1, 6, 14, 7, 15, 3, 6, 4, 14, 2];
        let span = Span::new(3, 5);
        let patch = editor.propose_patch(&tokens, span, &device);
        assert_eq!(patch.span, span);
        assert_eq!(patch.replacement.len(), 2);
        assert!(patch.replacement.iter().all(|&t| (t as usize) < 22));
    }
}
