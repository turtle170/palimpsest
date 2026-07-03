//! Phase 3: train the Critic to regress ablation-derived per-token
//! weights, and validate by rank correlation on held-out sequences.
//!
//! Checkpoint criterion: mean per-sequence Spearman correlation between
//! predicted and true (ablation) weights on held-out data, meaningfully
//! above a shuffled-prediction baseline.

use std::fs;
use std::path::PathBuf;

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use clap::Parser;
use palimpsest_critic::{Critic, CriticConfig};
use palimpsest_data::LabeledSequence;
use palimpsest_data::batch::BatchIndexer;
use palimpsest_data::{KvTaskConfig, TokenBatch};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::backend::{Inference, Train, device};
use crate::cmd::load_labels;
use crate::stats::spearman;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "artifacts/ablation")]
    pub labels_dir: PathBuf,
    #[arg(long, default_value = "artifacts/critic")]
    pub out_dir: PathBuf,
    #[arg(long, default_value_t = 2000)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 200)]
    pub log_every: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();
    let vocab_size = {
        use palimpsest_core::tokenizer::Tokenizer;
        task.vocab().vocab_size()
    };

    let train_set = load_labels(&args.labels_dir.join("train.jsonl"))?;
    let valid_set = load_labels(&args.labels_dir.join("valid.jsonl"))?;
    anyhow::ensure!(!train_set.is_empty() && !valid_set.is_empty(), "empty label sets");
    let seq_len = train_set[0].tokens.len();

    let config = CriticConfig::new(vocab_size).with_max_seq_len(seq_len);
    let mut model = config.init::<Train>(&device);
    let mut optim = AdamWConfig::new().init();
    let mut indexer = BatchIndexer::new(train_set.len(), args.batch_size, 30);

    println!(
        "Training Critic: {} steps on {} labeled sequences ({} held out)",
        args.steps,
        train_set.len(),
        valid_set.len()
    );

    for step in 1..=args.steps {
        let indices = indexer.next_batch().to_vec();
        let (tokens, targets, mask) = label_batch::<Train>(&train_set, &indices, &device);
        let preds = model.forward(tokens);
        // Masked MSE: positions never covered by a sampled span carry a
        // fake 0 label — exclude them rather than training on noise.
        let sq_err = (preds - targets).powf_scalar(2.0) * mask.clone();
        let loss = sq_err.sum() / mask.sum();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(args.lr, model, grads);

        if step % args.log_every == 0 || step == 1 {
            let train_loss: f32 = loss.into_scalar().elem();
            let rho = mean_spearman(&model.valid(), &valid_set, &device);
            println!("step {step:>5} | masked MSE {train_loss:.5} | held-out Spearman {rho:.3}");
        }
    }

    let inference_model = model.valid();
    let rho = mean_spearman(&inference_model, &valid_set, &device);
    let rho_random = shuffled_baseline(&valid_set, 123);
    println!("final held-out Spearman: {rho:.3} (shuffled baseline {rho_random:.3})");
    if rho < 0.5 {
        println!("WARNING: Critic rank correlation is weak; Phase 3 checkpoint NOT met.");
    }

    fs::create_dir_all(&args.out_dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    inference_model
        .save_file(args.out_dir.join("model"), &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint save failed: {e}"))?;
    config.save(args.out_dir.join("config.json"))?;
    println!("saved Critic to {}", args.out_dir.display());
    Ok(())
}

/// Build (tokens, target_weights, coverage_mask) tensors for a label batch.
fn label_batch<B: Backend>(
    labels: &[LabeledSequence],
    indices: &[usize],
    device: &B::Device,
) -> (
    burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    Tensor<B, 2>,
    Tensor<B, 2>,
) {
    let seqs: Vec<&[u32]> = indices.iter().map(|&i| labels[i].tokens.as_slice()).collect();
    let tokens = TokenBatch::<B>::from_sequences(&seqs, device).tokens;
    let t = labels[indices[0]].tokens.len();
    let flat_w: Vec<f32> = indices
        .iter()
        .flat_map(|&i| labels[i].token_weights.iter().copied())
        .collect();
    let flat_m: Vec<f32> = indices
        .iter()
        .flat_map(|&i| labels[i].covered.iter().map(|&c| c as u8 as f32))
        .collect();
    let targets = Tensor::from_data(TensorData::new(flat_w, [indices.len(), t]), device);
    let mask = Tensor::from_data(TensorData::new(flat_m, [indices.len(), t]), device);
    (tokens, targets, mask)
}

/// Mean per-sequence Spearman between Critic predictions and ablation
/// labels, over covered positions only.
fn mean_spearman(
    model: &Critic<Inference>,
    labels: &[LabeledSequence],
    device: &burn::tensor::Device<Inference>,
) -> f64 {
    let indices: Vec<usize> = (0..labels.len()).collect();
    let (tokens, _, _) = label_batch::<Inference>(labels, &indices, device);
    let preds = model.forward(tokens); // [n, t]
    let t = labels[0].tokens.len();
    let flat: Vec<f32> = preds.into_data().to_vec().unwrap();

    let mut sum = 0.0;
    let mut n = 0usize;
    for (i, lab) in labels.iter().enumerate() {
        let row = &flat[i * t..(i + 1) * t];
        let (p, y): (Vec<f32>, Vec<f32>) = lab
            .covered
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(j, _)| (row[j], lab.token_weights[j]))
            .unzip();
        if p.len() >= 3 {
            sum += spearman(&p, &y);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Reference point: Spearman of label weights against a random shuffle of
/// themselves — what "no skill" looks like on this exact label
/// distribution (≈ 0 in expectation).
fn shuffled_baseline(labels: &[LabeledSequence], seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum = 0.0;
    let mut n = 0usize;
    for lab in labels {
        let y: Vec<f32> = lab
            .covered
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(j, _)| lab.token_weights[j])
            .collect();
        if y.len() >= 3 {
            let mut shuffled = y.clone();
            shuffled.shuffle(&mut rng);
            sum += spearman(&shuffled, &y);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}
