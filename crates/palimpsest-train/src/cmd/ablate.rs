//! Phase 2: run the ablation labeling pipeline against a trained Drafter
//! and write labeled datasets for the Critic (Phase 3).
//!
//! Checkpoint criterion (printed at the end): ablating the *queried*
//! key-value pair must produce a much larger NLL delta than ablating
//! unqueried (filler) pairs.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use burn::module::Module;
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Int, Tensor, TensorData};
use clap::Parser;
use palimpsest_core::tensor_util::sequence_nll;
use palimpsest_core::tokenizer::TokenId;
use palimpsest_data::ablation::{AblationConfig, LabeledSequence, label_sequence};
use palimpsest_data::{KvExample, KvTaskConfig, generate_examples};
use palimpsest_drafter::{Drafter, DrafterConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::{Inference, device};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "artifacts/drafter")]
    pub drafter_dir: PathBuf,
    #[arg(long, default_value = "artifacts/ablation")]
    pub out_dir: PathBuf,
    #[arg(long, default_value_t = 2048)]
    pub train_sequences: usize,
    #[arg(long, default_value_t = 256)]
    pub valid_sequences: usize,
    /// Main cost/fidelity knob: ablation samples per sequence.
    #[arg(long, default_value_t = 16)]
    pub spans_per_sequence: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();

    let drafter_config = DrafterConfig::load(args.drafter_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("failed to load drafter config: {e}"))?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let drafter: Drafter<Inference> = drafter_config
        .init(&device)
        .load_file(args.drafter_dir.join("model"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("failed to load drafter checkpoint: {e}"))?;

    let ablation_config = AblationConfig {
        spans_per_sequence: args.spans_per_sequence,
        ..AblationConfig::default()
    };

    fs::create_dir_all(&args.out_dir)?;
    // Fresh seeds, disjoint from Drafter training seeds (1, 2).
    for (name, n, seed) in [
        ("train", args.train_sequences, 10u64),
        ("valid", args.valid_sequences, 11u64),
    ] {
        let examples = generate_examples(&task, n, seed);
        let labeled = label_examples(&drafter, &examples, &ablation_config, seed, &device);
        report_sanity(name, &examples, &labeled);

        let path = args.out_dir.join(format!("{name}.jsonl"));
        let mut file = fs::File::create(&path)?;
        for l in &labeled {
            writeln!(file, "{}", serde_json::to_string(l)?)?;
        }
        println!("wrote {} labeled sequences to {}", labeled.len(), path.display());
    }
    serde_json::to_writer_pretty(
        fs::File::create(args.out_dir.join("ablation_config.json"))?,
        &ablation_config,
    )?;
    Ok(())
}

/// Label a batch of examples. Each example's pair spans are appended as
/// deterministic "extra" spans — they are legitimate labeled spans for the
/// Critic *and* give us the queried-vs-filler sanity signal for free.
fn label_examples(
    drafter: &Drafter<Inference>,
    examples: &[KvExample],
    config: &AblationConfig,
    seed: u64,
    device: &burn::tensor::Device<Inference>,
) -> Vec<LabeledSequence> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7919).wrapping_add(config.seed));
    let scorer = |inputs: &[Vec<TokenId>], original: &[TokenId]| -> Vec<Vec<f32>> {
        nll_rows(drafter, inputs, original, device)
    };
    examples
        .iter()
        .map(|ex| label_sequence(&ex.tokens, &ex.pair_spans, &scorer, config, &mut rng))
        .collect()
}

/// Score a batch of (possibly masked) inputs against the original
/// continuation: returns per-prediction-index NLL rows.
fn nll_rows(
    drafter: &Drafter<Inference>,
    inputs: &[Vec<TokenId>],
    original: &[TokenId],
    device: &burn::tensor::Device<Inference>,
) -> Vec<Vec<f32>> {
    let n = inputs.len();
    let t = original.len();
    // Inputs: each variant truncated to t-1 (autoregressive input view).
    let flat_in: Vec<i64> = inputs
        .iter()
        .flat_map(|seq| seq[..t - 1].iter().map(|&x| x as i64))
        .collect();
    let input_tensor =
        Tensor::<Inference, 2, Int>::from_data(TensorData::new(flat_in, [n, t - 1]), device);
    // Targets: the ORIGINAL continuation for every variant.
    let flat_tgt: Vec<i64> = original[1..].iter().map(|&x| x as i64).collect();
    let target_row =
        Tensor::<Inference, 2, Int>::from_data(TensorData::new(flat_tgt, [1, t - 1]), device);
    let targets = target_row.expand([n, t - 1]);

    let logits = drafter.forward(input_tensor);
    let nll = sequence_nll(logits, targets); // [n, t-1]
    let flat: Vec<f32> = nll.into_data().to_vec().unwrap();
    flat.chunks(t - 1).map(|c| c.to_vec()).collect()
}

/// Print (and sanity-assert) the queried-pair vs filler-pair weight gap.
/// Relies on pair spans being the LAST `pair_spans.len()` entries of
/// span_weights (label_examples appends them as extra spans).
fn report_sanity(name: &str, examples: &[KvExample], labeled: &[LabeledSequence]) {
    let mut queried_sum = 0.0f64;
    let mut filler_sum = 0.0f64;
    let mut queried_n = 0usize;
    let mut filler_n = 0usize;
    for (ex, lab) in examples.iter().zip(labeled) {
        let n_pairs = ex.pair_spans.len();
        let pair_weights = &lab.span_weights[lab.span_weights.len() - n_pairs..];
        for (i, sw) in pair_weights.iter().enumerate() {
            debug_assert_eq!(sw.span, ex.pair_spans[i]);
            if i == ex.queried_pair {
                queried_sum += sw.weight as f64;
                queried_n += 1;
            } else {
                filler_sum += sw.weight as f64;
                filler_n += 1;
            }
        }
    }
    let queried_mean = queried_sum / queried_n.max(1) as f64;
    let filler_mean = filler_sum / filler_n.max(1) as f64;
    println!(
        "[{name}] sanity: mean ablation weight — queried pair {queried_mean:.4} vs filler pairs {filler_mean:.4} (ratio {:.1}x)",
        queried_mean / filler_mean.abs().max(1e-6)
    );
    if queried_mean < 2.0 * filler_mean.abs() {
        println!(
            "[{name}] WARNING: queried-pair weights are NOT clearly above filler — labels look unsound; do not train the Critic on this."
        );
    }
}
