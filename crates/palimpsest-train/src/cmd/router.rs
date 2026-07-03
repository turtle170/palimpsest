//! Phase 4: train the Router to predict Critic weights *cheaply* from the
//! Drafter's internal activations (one hidden layer), instead of running a
//! full Critic pass.
//!
//! Checkpoint criterion: on held-out sequences, the Router's top-k scored
//! tokens overlap the ablation-truth top-k tokens far above the shuffled
//! baseline.

use std::fs;
use std::path::PathBuf;

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{ElementConversion, Tensor};
use clap::Parser;
use palimpsest_critic::{Critic, CriticConfig};
use palimpsest_data::batch::BatchIndexer;
use palimpsest_data::{KvTaskConfig, LabeledSequence, generate_examples};
use palimpsest_drafter::{Drafter, DrafterConfig};
use palimpsest_router::{Router, RouterConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::backend::{Inference, Train, device};
use crate::cmd::{batch_tokens, load_labels};
use crate::stats::{spearman, top_k_overlap};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "artifacts/drafter")]
    pub drafter_dir: PathBuf,
    #[arg(long, default_value = "artifacts/critic")]
    pub critic_dir: PathBuf,
    /// Held-out ablation labels used as ground truth for the checkpoint.
    #[arg(long, default_value = "artifacts/ablation")]
    pub labels_dir: PathBuf,
    #[arg(long, default_value = "artifacts/router")]
    pub out_dir: PathBuf,
    /// Which Drafter block's output the Router reads (0-based). Default:
    /// last block. Which layer is most informative is an empirical
    /// question — this is the knob.
    #[arg(long)]
    pub layer: Option<usize>,
    #[arg(long, default_value_t = 1500)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 200)]
    pub log_every: usize,
    #[arg(long, default_value_t = 3)]
    pub top_k: usize,
}

/// Sidecar metadata: which layer the Router was trained against. The edit
/// loop must read the same layer.
#[derive(Serialize, Deserialize)]
pub struct RouterMeta {
    pub layer: usize,
    pub d_input: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let drafter_config = DrafterConfig::load(args.drafter_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load drafter config: {e}"))?;
    let drafter: Drafter<Inference> = drafter_config
        .init(&device)
        .load_file(args.drafter_dir.join("model"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("load drafter: {e}"))?;
    let critic_config = CriticConfig::load(args.critic_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load critic config: {e}"))?;
    let critic: Critic<Inference> = critic_config
        .init(&device)
        .load_file(args.critic_dir.join("model"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("load critic: {e}"))?;

    let layer = args.layer.unwrap_or(drafter_config.n_layers - 1);
    anyhow::ensure!(layer < drafter_config.n_layers, "layer out of range");

    // Fresh sequences for Router training (Critic provides dense targets,
    // so no ablation pass is needed here — that's the whole point).
    let train_set = generate_examples(&task, 2048, 20);
    let valid_labels = load_labels(&args.labels_dir.join("valid.jsonl"))?;

    let router_config = RouterConfig::new(drafter_config.d_model);
    let mut router = router_config.init::<Train>(&device);
    let mut optim = AdamWConfig::new().init();
    let mut indexer = BatchIndexer::new(train_set.len(), args.batch_size, 50);

    println!(
        "Training Router on Drafter layer {layer} activations: {} steps",
        args.steps
    );

    for step in 1..=args.steps {
        let indices = indexer.next_batch().to_vec();
        let batch = batch_tokens::<Inference>(&train_set, &indices, &device);
        // Drafter + Critic run without autodiff; only the Router trains.
        let (_, hidden) = drafter.forward_with_hidden(batch.tokens.clone());
        let targets = critic.forward(batch.tokens);
        let hidden_ad = Tensor::<Train, 3>::from_inner(hidden[layer].clone());
        let targets_ad = Tensor::<Train, 2>::from_inner(targets);

        let preds = router.forward(hidden_ad);
        let loss = (preds - targets_ad).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &router);
        router = optim.step(args.lr, router, grads);

        if step % args.log_every == 0 || step == 1 {
            let mse: f32 = loss.into_scalar().elem();
            println!("step {step:>5} | MSE vs Critic {mse:.5}");
        }
    }

    // Checkpoint eval: Router scores vs ablation ground truth (held-out).
    let router_inf = router.valid();
    let (overlap, overlap_shuffled, rho) = evaluate_against_ablation(
        &drafter,
        &router_inf,
        layer,
        &valid_labels,
        args.top_k,
        &device,
    );
    println!(
        "held-out vs ablation truth: top-{} overlap {overlap:.3} (shuffled baseline {overlap_shuffled:.3}), Spearman {rho:.3}",
        args.top_k
    );
    if overlap < overlap_shuffled + 0.2 {
        println!("WARNING: Router overlap barely above baseline; Phase 4 checkpoint NOT met.");
    }

    fs::create_dir_all(&args.out_dir)?;
    router_inf
        .save_file(args.out_dir.join("model"), &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint save failed: {e}"))?;
    router_config.save(args.out_dir.join("config.json"))?;
    serde_json::to_writer_pretty(
        fs::File::create(args.out_dir.join("meta.json"))?,
        &RouterMeta {
            layer,
            d_input: drafter_config.d_model,
        },
    )?;
    println!("saved Router to {}", args.out_dir.display());
    Ok(())
}

/// Router scores for one sequence from Drafter activations — shared with
/// the Phase 6 loop.
pub fn router_scores_for(
    drafter: &Drafter<Inference>,
    router: &Router<Inference>,
    layer: usize,
    tokens: &[u32],
    device: &burn::tensor::Device<Inference>,
) -> Vec<f32> {
    let batch = palimpsest_data::TokenBatch::<Inference>::from_sequences(&[tokens], device);
    let (_, hidden) = drafter.forward_with_hidden(batch.tokens);
    let scores = router.forward(hidden[layer].clone());
    scores.into_data().to_vec().unwrap()
}

fn evaluate_against_ablation(
    drafter: &Drafter<Inference>,
    router: &Router<Inference>,
    layer: usize,
    labels: &[LabeledSequence],
    k: usize,
    device: &burn::tensor::Device<Inference>,
) -> (f64, f64, f64) {
    let mut rng = StdRng::seed_from_u64(99);
    let mut overlap_sum = 0.0;
    let mut shuffled_sum = 0.0;
    let mut rho_sum = 0.0;
    let mut n = 0usize;
    for lab in labels {
        let scores = router_scores_for(drafter, router, layer, &lab.tokens, device);
        // Compare only on positions the ablation pass actually covered.
        let (pred, truth): (Vec<f32>, Vec<f32>) = lab
            .covered
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(j, _)| (scores[j], lab.token_weights[j]))
            .unzip();
        if pred.len() < k.max(3) {
            continue;
        }
        overlap_sum += top_k_overlap(&pred, &truth, k);
        let mut shuffled = pred.clone();
        shuffled.shuffle(&mut rng);
        shuffled_sum += top_k_overlap(&shuffled, &truth, k);
        rho_sum += spearman(&pred, &truth);
        n += 1;
    }
    let n = n.max(1) as f64;
    (overlap_sum / n, shuffled_sum / n, rho_sum / n)
}
