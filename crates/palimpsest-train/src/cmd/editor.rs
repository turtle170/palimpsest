//! Phase 5: train the Editor on synthetic corrupt-and-repair pairs.
//!
//! Training corruption: replace a random span with random *content* tokens
//! (keys/values). Held-out evaluation uses MASK-fill corruption — a
//! corruption pattern the Editor never saw — plus fresh sequences.
//!
//! Checkpoint criteria: on unseen corruptions, applying the Editor's patch
//! (a) reconstructs the original span far above chance, and (b) brings the
//! Drafter's NLL of the sequence back down toward the clean value.
//!
//! OPEN QUESTION (per plan): corrupt-and-repair is only the baseline
//! objective. It teaches "restore what plausibly belongs here", not
//! "improve a weak draft"; the right objective past this baseline is
//! unresolved.

use std::fs;
use std::path::PathBuf;

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use clap::Parser;
use palimpsest_core::span::Span;
use palimpsest_core::tensor_util::{mean_nll, sequence_nll};
use palimpsest_core::tokenizer::{TokenId, Tokenizer};
use palimpsest_data::batch::BatchIndexer;
use palimpsest_data::vocab::{self, KvVocab};
use palimpsest_data::{KvExample, KvTaskConfig, TokenBatch, generate_examples};
use palimpsest_drafter::{Drafter, DrafterConfig};
use palimpsest_editor::{Editor, EditorConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::backend::{Inference, Train, device};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "artifacts/drafter")]
    pub drafter_dir: PathBuf,
    #[arg(long, default_value = "artifacts/editor")]
    pub out_dir: PathBuf,
    #[arg(long, default_value_t = 3000)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 200)]
    pub log_every: usize,
    #[arg(long, default_value_t = 1)]
    pub min_span_len: usize,
    #[arg(long, default_value_t = 3)]
    pub max_span_len: usize,
}

/// How a span gets corrupted.
#[derive(Clone, Copy)]
pub enum Corruption {
    /// Random key/value content tokens (training distribution).
    RandomContent,
    /// MASK fill (held out from training — "unseen corruption pattern").
    MaskFill,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();
    let vocab = task.vocab();

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let drafter_config = DrafterConfig::load(args.drafter_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load drafter config: {e}"))?;
    let drafter: Drafter<Inference> = drafter_config
        .init(&device)
        .load_file(args.drafter_dir.join("model"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("load drafter: {e}"))?;

    let train_set = generate_examples(&task, 4096, 40);
    let valid_set = generate_examples(&task, 512, 41);

    let config = EditorConfig::new(vocab.vocab_size()).with_max_seq_len(task.seq_len());
    let mut model = config.init::<Train>(&device);
    let mut optim = AdamWConfig::new().init();
    let mut indexer = BatchIndexer::new(train_set.len(), args.batch_size, 60);
    let mut rng = StdRng::seed_from_u64(61);

    println!("Training Editor (corrupt-and-repair): {} steps", args.steps);

    for step in 1..=args.steps {
        let indices = indexer.next_batch().to_vec();
        let mut corrupted: Vec<Vec<TokenId>> = Vec::with_capacity(indices.len());
        let mut spans: Vec<Span> = Vec::with_capacity(indices.len());
        for &i in &indices {
            let ex = &train_set[i];
            let (c, s) = corrupt(
                &ex.tokens,
                &vocab,
                Corruption::RandomContent,
                args.min_span_len..=args.max_span_len,
                &mut rng,
            );
            corrupted.push(c);
            spans.push(s);
        }
        let originals: Vec<&[TokenId]> =
            indices.iter().map(|&i| train_set[i].tokens.as_slice()).collect();
        let (loss, _) = editor_loss::<Train>(&model, &corrupted, &spans, &originals, &device);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(args.lr, model, grads);

        if step % args.log_every == 0 || step == 1 {
            let l: f32 = loss.into_scalar().elem();
            println!("step {step:>5} | span CE loss {l:.4}");
        }
    }

    // ---- Held-out evaluation on the UNSEEN corruption pattern ----
    let editor_inf = model.valid();
    let report = evaluate(
        &editor_inf,
        &drafter,
        &valid_set,
        &vocab,
        Corruption::MaskFill,
        args.min_span_len..=args.max_span_len,
        &device,
        &mut StdRng::seed_from_u64(62),
    );
    println!(
        "held-out (MASK corruption): span reconstruction {:.3} | drafter NLL clean {:.4} / corrupted {:.4} / repaired {:.4}",
        report.reconstruction_rate, report.nll_clean, report.nll_corrupted, report.nll_repaired
    );
    if report.nll_repaired > (report.nll_clean + report.nll_corrupted) / 2.0 {
        println!("WARNING: repairs recover less than half the NLL damage; Phase 5 checkpoint NOT met.");
    }

    fs::create_dir_all(&args.out_dir)?;
    editor_inf
        .save_file(args.out_dir.join("model"), &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint save failed: {e}"))?;
    config.save(args.out_dir.join("config.json"))?;
    println!("saved Editor to {}", args.out_dir.display());
    Ok(())
}

/// Corrupt one random span; returns (corrupted tokens, span).
/// Spans live in `[1, len-1)` — BOS and the final EOS stay intact.
pub fn corrupt(
    tokens: &[TokenId],
    vocab: &KvVocab,
    kind: Corruption,
    span_lens: std::ops::RangeInclusive<usize>,
    rng: &mut StdRng,
) -> (Vec<TokenId>, Span) {
    let lo = 1usize;
    let hi = tokens.len() - 1;
    let span_len = rng.random_range(span_lens).min(hi - lo);
    let start = rng.random_range(lo..=hi - span_len);
    let span = Span::new(start, start + span_len);
    let mut out = tokens.to_vec();
    for p in span.positions() {
        out[p] = match kind {
            Corruption::RandomContent => {
                let content_lo = vocab::NUM_SPECIAL;
                let content_hi = vocab.vocab_size() as TokenId;
                rng.random_range(content_lo..content_hi)
            }
            Corruption::MaskFill => vocab::MASK,
        };
    }
    (out, span)
}

/// Cross-entropy on in-span positions only. Returns (loss, span_mask).
fn editor_loss<B: Backend>(
    model: &Editor<B>,
    corrupted: &[Vec<TokenId>],
    spans: &[Span],
    originals: &[&[TokenId]],
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let n = corrupted.len();
    let t = corrupted[0].len();
    let corr_refs: Vec<&[TokenId]> = corrupted.iter().map(|c| c.as_slice()).collect();
    let inputs = TokenBatch::<B>::from_sequences(&corr_refs, device).tokens;
    let targets = TokenBatch::<B>::from_sequences(originals, device).tokens;

    let flat_mask: Vec<i64> = corrupted
        .iter()
        .zip(spans)
        .flat_map(|(_, s)| (0..t).map(move |p| s.contains(p) as i64))
        .collect();
    let span_mask_int =
        Tensor::<B, 2, Int>::from_data(TensorData::new(flat_mask, [n, t]), device);
    let span_mask = span_mask_int.clone().float();

    let logits = model.forward(inputs, span_mask_int);
    let nll = sequence_nll(logits, targets); // [n, t] vs original tokens
    let loss = (nll * span_mask.clone()).sum() / span_mask.clone().sum();
    (loss, span_mask)
}

pub struct EvalReport {
    pub reconstruction_rate: f64,
    pub nll_clean: f32,
    pub nll_corrupted: f32,
    pub nll_repaired: f32,
}

/// Corrupt each valid example, repair with the Editor, and measure both
/// exact reconstruction and Drafter-NLL recovery.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    editor: &Editor<Inference>,
    drafter: &Drafter<Inference>,
    examples: &[KvExample],
    vocab: &KvVocab,
    kind: Corruption,
    span_lens: std::ops::RangeInclusive<usize>,
    device: &burn::tensor::Device<Inference>,
    rng: &mut StdRng,
) -> EvalReport {
    let mut clean: Vec<&[TokenId]> = Vec::new();
    let mut corrupted: Vec<Vec<TokenId>> = Vec::new();
    let mut repaired: Vec<Vec<TokenId>> = Vec::new();
    let mut reconstructed = 0usize;

    for ex in examples {
        let (c, span) = corrupt(&ex.tokens, vocab, kind, span_lens.clone(), rng);
        let patch = editor.propose_patch(&c, span, device);
        let r = patch.apply(&c);
        if r[span.start..span.end] == ex.tokens[span.start..span.end] {
            reconstructed += 1;
        }
        clean.push(&ex.tokens);
        corrupted.push(c);
        repaired.push(r);
    }

    let nll_of = |seqs: &[&[TokenId]]| -> f32 {
        let batch = TokenBatch::<Inference>::from_sequences(seqs, device);
        let (inputs, targets) = batch.autoregressive_views();
        let logits = drafter.forward(inputs);
        mean_nll(logits, targets).into_scalar().elem()
    };
    let corr_refs: Vec<&[TokenId]> = corrupted.iter().map(|c| c.as_slice()).collect();
    let rep_refs: Vec<&[TokenId]> = repaired.iter().map(|c| c.as_slice()).collect();

    EvalReport {
        reconstruction_rate: reconstructed as f64 / examples.len() as f64,
        nll_clean: nll_of(&clean),
        nll_corrupted: nll_of(&corr_refs),
        nll_repaired: nll_of(&rep_refs),
    }
}
