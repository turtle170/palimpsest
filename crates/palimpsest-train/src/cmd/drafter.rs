//! Phase 1: train the Drafter standalone with a next-token loss.
//!
//! Checkpoint criteria: loss decreases, answer-position accuracy climbs
//! well above chance (1/num_values = 12.5% at defaults), and the model
//! round-trips through save/load.

use std::fs;
use std::path::PathBuf;

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::ElementConversion;
use clap::Parser;
use palimpsest_core::tensor_util::mean_nll;
use palimpsest_core::tokenizer::Tokenizer;
use palimpsest_data::batch::BatchIndexer;
use palimpsest_data::{KvTaskConfig, generate_examples};
use palimpsest_drafter::DrafterConfig;

use crate::backend::{Inference, Train, device};
use crate::cmd::{answer_accuracy, batch_tokens};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value_t = 3000)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 4096)]
    pub train_examples: usize,
    #[arg(long, default_value_t = 512)]
    pub valid_examples: usize,
    #[arg(long, default_value = "artifacts/drafter")]
    pub out_dir: PathBuf,
    /// Log/eval interval in steps.
    #[arg(long, default_value_t = 200)]
    pub log_every: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();
    let vocab = task.vocab();

    // Different seeds for train/valid; see kv_task.rs on split contamination.
    let train_set = generate_examples(&task, args.train_examples, 1);
    let valid_set = generate_examples(&task, args.valid_examples, 2);
    let answer_pos = task.answer_pos();

    let model_config = DrafterConfig::new(vocab.vocab_size()).with_max_seq_len(task.seq_len());
    let mut model = model_config.init::<Train>(&device);
    let mut optim = AdamWConfig::new().init();
    let mut indexer = BatchIndexer::new(train_set.len(), args.batch_size, 3);

    println!(
        "Training Drafter: {} steps, batch {}, lr {}, vocab {}, seq_len {}",
        args.steps,
        args.batch_size,
        args.lr,
        vocab.vocab_size(),
        task.seq_len()
    );

    for step in 1..=args.steps {
        let indices = indexer.next_batch().to_vec();
        let batch = batch_tokens::<Train>(&train_set, &indices, &device);
        let (inputs, targets) = batch.autoregressive_views();
        let logits = model.forward(inputs);
        let loss = mean_nll(logits, targets);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(args.lr, model, grads);

        if step % args.log_every == 0 || step == 1 {
            let train_loss: f32 = loss.into_scalar().elem();
            let (valid_loss, valid_acc) = evaluate(&model.valid(), &valid_set, answer_pos, &device);
            println!(
                "step {step:>5} | train loss {train_loss:.4} | valid loss {valid_loss:.4} | answer acc {valid_acc:.3}"
            );
        }
    }

    // Final eval + checkpoint.
    let inference_model = model.valid();
    let (valid_loss, valid_acc) = evaluate(&inference_model, &valid_set, answer_pos, &device);
    println!("final: valid loss {valid_loss:.4} | answer acc {valid_acc:.3}");

    fs::create_dir_all(&args.out_dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    inference_model
        .clone()
        .save_file(args.out_dir.join("model"), &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint save failed: {e}"))?;
    model_config.save(args.out_dir.join("config.json"))?;

    // Round-trip the checkpoint to prove save/load works (Phase 1
    // checkpoint requirement), and that the reloaded model agrees.
    let reloaded = model_config
        .init::<Inference>(&device)
        .load_file(args.out_dir.join("model"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("checkpoint load failed: {e}"))?;
    let (reload_loss, reload_acc) = evaluate(&reloaded, &valid_set, answer_pos, &device);
    println!("reloaded checkpoint: valid loss {reload_loss:.4} | answer acc {reload_acc:.3}");
    anyhow::ensure!(
        (reload_loss - valid_loss).abs() < 1e-4,
        "reloaded model disagrees with trained model"
    );

    println!("saved Drafter to {}", args.out_dir.display());
    Ok(())
}

fn evaluate(
    model: &palimpsest_drafter::Drafter<Inference>,
    examples: &[palimpsest_data::KvExample],
    answer_pos: usize,
    device: &burn::tensor::Device<Inference>,
) -> (f32, f32) {
    let indices: Vec<usize> = (0..examples.len()).collect();
    let batch = batch_tokens::<Inference>(examples, &indices, device);
    let (inputs, targets) = batch.autoregressive_views();
    let logits = model.forward(inputs);
    let loss: f32 = mean_nll(logits.clone(), targets.clone())
        .into_scalar()
        .elem();
    let acc = answer_accuracy(logits, targets, answer_pos);
    (loss, acc)
}
