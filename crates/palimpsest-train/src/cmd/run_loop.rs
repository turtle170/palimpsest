//! Phase 6: the full self-editing loop.
//!
//! Drafter produces (or is handed) a scratchpad → Router flags spans from
//! Drafter activations → Critic weighs flagged spans and decides halt →
//! Editor patches the heaviest span → repeat, under a hard iteration cap.
//!
//! Two evaluation modes, both reported:
//!  * `corrupted`: clean sequences with 1–2 randomly corrupted spans stand
//!    in for imperfect drafts; the loop should repair them.
//!  * `organic`: the Drafter greedily completes the answer from the query
//!    prefix; with a well-trained Drafter these are mostly already correct,
//!    so this mode mainly demonstrates the halting signal doesn't fire
//!    edits spuriously.
//!
//! Every trajectory (draft → patch → … → final) is dumped as JSONL for
//! inspection — degenerate behaviors (oscillation, no-op loops) are only
//! findable by reading trajectories.

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use burn::module::Module;
use burn::prelude::Config as _;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use clap::Parser;
use palimpsest_core::scratchpad::ScratchpadState;
use palimpsest_core::span::{Span, merge_positions_into_spans};
use palimpsest_core::tensor_util::mean_nll;
use palimpsest_core::tokenizer::{TokenId, Tokenizer};
use palimpsest_critic::{Critic, CriticConfig, HaltingConfig, span_weight};
use palimpsest_data::vocab::{self, KvVocab};
use palimpsest_data::{KvTaskConfig, TokenBatch, generate_examples};
use palimpsest_drafter::{Drafter, DrafterConfig};
use palimpsest_editor::{Editor, EditorConfig};
use palimpsest_router::{Router, RouterConfig, flag_positions};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::backend::{Inference, device};
use crate::cmd::editor::{Corruption, corrupt};
use crate::cmd::router::{RouterMeta, router_scores_for};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "artifacts/drafter")]
    pub drafter_dir: PathBuf,
    #[arg(long, default_value = "artifacts/router")]
    pub router_dir: PathBuf,
    #[arg(long, default_value = "artifacts/critic")]
    pub critic_dir: PathBuf,
    #[arg(long, default_value = "artifacts/editor")]
    pub editor_dir: PathBuf,
    #[arg(long, default_value = "artifacts/loop")]
    pub out_dir: PathBuf,
    #[arg(long, default_value_t = 200)]
    pub num_examples: usize,
    /// Hard safety cap on edit iterations regardless of the Critic's
    /// halting decision (the Critic may be unreliable early on).
    #[arg(long, default_value_t = 8)]
    pub max_steps: usize,
    /// Router score above which a position is flagged for editing.
    #[arg(long, default_value_t = 0.2)]
    pub router_threshold: f32,
    /// Critic span weight below which remaining spans are not worth
    /// editing (halting signal).
    #[arg(long, default_value_t = 0.1)]
    pub halt_threshold: f64,
    #[arg(long, default_value_t = 3)]
    pub max_span_len: usize,
    /// Corrupted-draft mode: how many spans to corrupt per example.
    #[arg(long, default_value_t = 1)]
    pub corrupt_spans: usize,
}

struct Models {
    drafter: Drafter<Inference>,
    router: Router<Inference>,
    router_layer: usize,
    critic: Critic<Inference>,
    editor: Editor<Inference>,
}

#[derive(Serialize)]
struct Trajectory {
    initial: Vec<TokenId>,
    initial_readable: String,
    final_tokens: Vec<TokenId>,
    final_readable: String,
    edits: Vec<EditLog>,
    halted: bool,
    steps_used: usize,
    initial_consistent: bool,
    final_consistent: bool,
    exact_match_original: Option<bool>,
}

#[derive(Serialize)]
struct EditLog {
    step: usize,
    span: Span,
    before: Vec<TokenId>,
    after: Vec<TokenId>,
    router_score: f32,
    critic_weight: f32,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let device = device();
    let task = KvTaskConfig::default();
    let vocab = task.vocab();
    let models = load_models(&args, &device)?;
    let halting = HaltingConfig::new().with_weight_threshold(args.halt_threshold);
    fs::create_dir_all(&args.out_dir)?;

    // ---- Mode 1: corrupted drafts ----
    let examples = generate_examples(&task, args.num_examples, 70);
    let mut rng = StdRng::seed_from_u64(71);
    let mut trajectories = Vec::new();
    for ex in &examples {
        let mut draft = ex.tokens.clone();
        for _ in 0..args.corrupt_spans {
            let (c, _) = corrupt(&draft, &vocab, Corruption::RandomContent, 1..=2, &mut rng);
            draft = c;
        }
        let traj = edit_loop(&models, &halting, &args, &task, &vocab, draft, Some(&ex.tokens), &device);
        trajectories.push(traj);
    }
    report("corrupted", &task, &trajectories, &models, &device);
    write_trajectories(&args.out_dir.join("corrupted.jsonl"), &trajectories)?;

    // ---- Mode 2: organic drafts (Drafter greedy completion) ----
    let examples = generate_examples(&task, args.num_examples, 72);
    let mut trajectories = Vec::new();
    for ex in &examples {
        // Prefix ends right after QUERY; Drafter writes answer + EOS.
        let prefix = &ex.tokens[..ex.answer_pos];
        let draft = models
            .drafter
            .generate_greedy(prefix, ex.tokens.len() - ex.answer_pos, &device);
        let traj = edit_loop(&models, &halting, &args, &task, &vocab, draft, Some(&ex.tokens), &device);
        trajectories.push(traj);
    }
    report("organic", &task, &trajectories, &models, &device);
    write_trajectories(&args.out_dir.join("organic.jsonl"), &trajectories)?;

    Ok(())
}

/// One full edit loop over a single scratchpad.
#[allow(clippy::too_many_arguments)]
fn edit_loop(
    models: &Models,
    halting: &HaltingConfig,
    args: &Args,
    task: &KvTaskConfig,
    vocab: &KvVocab,
    draft: Vec<TokenId>,
    original: Option<&[TokenId]>,
    device: &burn::tensor::Device<Inference>,
) -> Trajectory {
    let initial_consistent = task_consistent(task, vocab, &draft);
    let mut pad = ScratchpadState::new(draft);
    // Spans already inspected on identical content — prevents oscillation
    // and re-editing stable spans; a span becomes eligible again only if
    // some edit changed its content.
    let mut visited: HashSet<(usize, usize, Vec<TokenId>)> = HashSet::new();
    let mut edits = Vec::new();
    let mut halted = false;
    let mut steps_used = 0;

    for step in 0..args.max_steps {
        steps_used = step + 1;
        let scores = router_scores_for(
            &models.drafter,
            &models.router,
            models.router_layer,
            &pad.tokens,
            device,
        );
        let flagged = flag_positions(&scores, args.router_threshold);
        let spans = merge_positions_into_spans(flagged, args.max_span_len);
        let weights = critic_weights(&models.critic, &pad.tokens, device);

        let candidates: Vec<(Span, f32)> = spans
            .into_iter()
            .filter(|s| {
                !visited.contains(&(s.start, s.end, pad.tokens[s.start..s.end].to_vec()))
            })
            .map(|s| (s, span_weight(&weights, s)))
            .collect();

        let remaining: Vec<f32> = candidates.iter().map(|&(_, w)| w).collect();
        if halting.should_halt(&remaining) {
            halted = true;
            steps_used = step; // this step performed no edit
            break;
        }

        let &(span, weight) = candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty since halting did not fire");
        visited.insert((span.start, span.end, pad.tokens[span.start..span.end].to_vec()));

        let patch = models.editor.propose_patch(&pad.tokens, span, device);
        if patch.is_noop_on(&pad.tokens) {
            continue; // span looks fine to the Editor; try next-heaviest next iter
        }
        let router_score = {
            let s = &scores[span.start..span.end];
            s.iter().sum::<f32>() / s.len() as f32
        };
        let before = pad.tokens[span.start..span.end].to_vec();
        pad.apply(patch.clone(), router_score, weight);
        edits.push(EditLog {
            step,
            span,
            before,
            after: patch.replacement,
            router_score,
            critic_weight: weight,
        });
    }

    Trajectory {
        initial_readable: vocab.decode(&pad.initial_draft),
        final_readable: vocab.decode(&pad.tokens),
        initial: pad.initial_draft.clone(),
        final_consistent: task_consistent(task, vocab, &pad.tokens),
        exact_match_original: original.map(|o| o == pad.tokens.as_slice()),
        final_tokens: pad.tokens,
        edits,
        halted,
        steps_used,
        initial_consistent,
    }
}

fn critic_weights(
    critic: &Critic<Inference>,
    tokens: &[TokenId],
    device: &burn::tensor::Device<Inference>,
) -> Vec<f32> {
    let batch = TokenBatch::<Inference>::from_sequences(&[tokens], device);
    critic.forward(batch.tokens).into_data().to_vec().unwrap()
}

/// Task-level correctness of a scratchpad, judged purely from task
/// structure (no model involved): layout tokens intact, all pairs
/// well-formed, the query key present among the pairs, and the answer
/// token equal to the queried key's bound value.
fn task_consistent(task: &KvTaskConfig, vocab: &KvVocab, tokens: &[TokenId]) -> bool {
    if tokens.len() != task.seq_len() {
        return false;
    }
    let sep_pos = 1 + 2 * task.num_pairs;
    let structure_ok = tokens[0] == vocab::BOS
        && tokens[sep_pos] == vocab::SEP
        && tokens[sep_pos + 2] == vocab::QUERY
        && tokens[task.seq_len() - 1] == vocab::EOS;
    if !structure_ok {
        return false;
    }
    let mut binding: Option<TokenId> = None;
    for i in 0..task.num_pairs {
        let k = tokens[1 + 2 * i];
        let v = tokens[2 + 2 * i];
        if !vocab.is_key(k) || !vocab.is_value(v) {
            return false;
        }
        if k == tokens[sep_pos + 1] {
            binding = Some(v);
        }
    }
    match binding {
        Some(v) => tokens[task.answer_pos()] == v,
        None => false, // query key not bound by any pair
    }
}

fn report(
    mode: &str,
    task: &KvTaskConfig,
    trajectories: &[Trajectory],
    models: &Models,
    device: &burn::tensor::Device<Inference>,
) {
    let n = trajectories.len() as f64;
    let rate = |f: &dyn Fn(&Trajectory) -> bool| {
        trajectories.iter().filter(|t| f(t)).count() as f64 / n
    };
    let consistent_before = rate(&|t| t.initial_consistent);
    let consistent_after = rate(&|t| t.final_consistent);
    let halted = rate(&|t| t.halted);
    let exact = rate(&|t| t.exact_match_original == Some(true));
    let mean_edits =
        trajectories.iter().map(|t| t.edits.len()).sum::<usize>() as f64 / n;
    let mean_steps = trajectories.iter().map(|t| t.steps_used).sum::<usize>() as f64 / n;

    let nll = |pick: &dyn Fn(&Trajectory) -> &[TokenId]| -> f32 {
        use burn::tensor::ElementConversion;
        let seqs: Vec<&[TokenId]> = trajectories.iter().map(|t| pick(t)).collect();
        let batch = TokenBatch::<Inference>::from_sequences(&seqs, device);
        let (inputs, targets) = batch.autoregressive_views();
        mean_nll(models.drafter.forward(inputs), targets)
            .into_scalar()
            .elem()
    };
    let nll_before = nll(&|t| &t.initial);
    let nll_after = nll(&|t| &t.final_tokens);

    let _ = task;
    println!(
        "[{mode}] n={} | consistent {:.3} -> {:.3} | exact-match {:.3} | drafter NLL {:.4} -> {:.4}",
        trajectories.len(),
        consistent_before,
        consistent_after,
        exact,
        nll_before,
        nll_after
    );
    println!(
        "[{mode}] halted within budget {:.3} | mean edit steps {:.2} | mean applied edits {:.2}",
        halted, mean_steps, mean_edits
    );
}

fn write_trajectories(path: &PathBuf, trajectories: &[Trajectory]) -> anyhow::Result<()> {
    let mut file = fs::File::create(path)?;
    for t in trajectories {
        writeln!(file, "{}", serde_json::to_string(t)?)?;
    }
    println!("wrote {} trajectories to {}", trajectories.len(), path.display());
    Ok(())
}

fn load_models(args: &Args, device: &burn::tensor::Device<Inference>) -> anyhow::Result<Models> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let drafter_config = DrafterConfig::load(args.drafter_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load drafter config: {e}"))?;
    let drafter = drafter_config
        .init(device)
        .load_file(args.drafter_dir.join("model"), &recorder, device)
        .map_err(|e| anyhow::anyhow!("load drafter: {e}"))?;
    let router_config = RouterConfig::load(args.router_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load router config: {e}"))?;
    let router = router_config
        .init(device)
        .load_file(args.router_dir.join("model"), &recorder, device)
        .map_err(|e| anyhow::anyhow!("load router: {e}"))?;
    let meta: RouterMeta =
        serde_json::from_reader(fs::File::open(args.router_dir.join("meta.json"))?)?;
    let critic_config = CriticConfig::load(args.critic_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load critic config: {e}"))?;
    let critic = critic_config
        .init(device)
        .load_file(args.critic_dir.join("model"), &recorder, device)
        .map_err(|e| anyhow::anyhow!("load critic: {e}"))?;
    let editor_config = EditorConfig::load(args.editor_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("load editor config: {e}"))?;
    let editor = editor_config
        .init(device)
        .load_file(args.editor_dir.join("model"), &recorder, device)
        .map_err(|e| anyhow::anyhow!("load editor: {e}"))?;
    Ok(Models {
        drafter,
        router,
        router_layer: meta.layer,
        critic,
        editor,
    })
}
