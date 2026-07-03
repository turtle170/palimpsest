//! Small reusable NN building blocks.
//!
//! Drafter, Critic, and Editor are all tiny transformers differing only in
//! masking (causal vs bidirectional) and heads, so the block lives here
//! once. NOTE: this is *code* reuse only — whether Router and Critic should
//! share learned weights/backbone is an open design question (currently:
//! separate models, per the project plan).

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// Pre-norm transformer block: LN → MHA → residual, LN → FFN → residual.
#[derive(Config, Debug)]
pub struct EncoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm_attn: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    norm_ffn: LayerNorm<B>,
    ffn_in: Linear<B>,
    ffn_out: Linear<B>,
    act: Gelu,
}

impl EncoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            norm_attn: LayerNormConfig::new(self.d_model).init(device),
            attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.dropout)
                .init(device),
            norm_ffn: LayerNormConfig::new(self.d_model).init(device),
            ffn_in: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ffn_out: LinearConfig::new(self.d_ff, self.d_model).init(device),
            act: Gelu::new(),
        }
    }
}

impl<B: Backend> EncoderBlock<B> {
    /// `mask`: `Some(causal mask)` for autoregressive use, `None` for
    /// bidirectional attention.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        let h = self.norm_attn.forward(x.clone());
        let mut input = MhaInput::self_attn(h);
        if let Some(m) = mask {
            input = input.mask_attn(m);
        }
        let x = x + self.attn.forward(input).context;
        let h = self.norm_ffn.forward(x.clone());
        let h = self.ffn_out.forward(self.act.forward(self.ffn_in.forward(h)));
        x + h
    }
}

/// Token embedding + learned positional embedding.
#[derive(Config, Debug)]
pub struct SeqEmbeddingConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

#[derive(Module, Debug)]
pub struct SeqEmbedding<B: Backend> {
    tok: Embedding<B>,
    pos: Embedding<B>,
}

impl SeqEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SeqEmbedding<B> {
        SeqEmbedding {
            tok: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            pos: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
        }
    }
}

impl<B: Backend> SeqEmbedding<B> {
    /// tokens `[batch, seq]` → embeddings `[batch, seq, d_model]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = tokens.dims();
        let device = tokens.device();
        let positions = Tensor::<B, 1, Int>::arange(0..t as i64, &device)
            .reshape([1, t])
            .expand([b, t]);
        self.tok.forward(tokens) + self.pos.forward(positions)
    }
}
