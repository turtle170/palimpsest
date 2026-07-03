//! Phase 2: ablation/perturbation labeling — the ground-truth causal
//! "weight" signal the Critic learns to imitate.
//!
//! For a sequence and a sampled span, we mask the span in the *input*,
//! rerun the model, and measure how much the per-position NLL of the
//! *original* tokens changes. Spans whose masking wrecks downstream
//! predictions get high weight.
//!
//! Decoupling: this module never sees a model. The caller supplies a
//! batched NLL function (`sequences -> per-prediction-index NLL rows`),
//! so the pipeline is testable with a fake scorer and reusable if the
//! Drafter changes.
//!
//! KNOWN BLIND SPOT (flagged, not silently resolved): input-ablation
//! weight measures how much a token matters *for predicting other
//! tokens*. A wrong token at a position nothing depends on (e.g. the
//! final answer token itself — only EOS follows it) gets ~zero weight
//! even though it is exactly what an editor should fix. "Importance"
//! and "badness" are different quantities; the plan derives labels from
//! ablation importance, so that is what this implements. See the Phase 6
//! evaluation for where this bites.

use palimpsest_core::span::Span;
use palimpsest_core::tokenizer::TokenId;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Which NLL deltas count toward a span's weight.
///
/// Prediction index `i` consumes input tokens `0..=i` and predicts token
/// `i+1`; masking span `[s, e)` perturbs predictions with `i >= s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AblationMetric {
    /// Mean delta over predictions of tokens strictly *after* the span
    /// (`i >= e-1`): "how much does this span matter for the original
    /// continuation". Matches the plan's default framing.
    Downstream,
    /// Mean delta over *all* perturbed predictions (`i >= s`), including
    /// re-predicting the span's own original tokens from perturbed
    /// context.
    FromSpanStart,
}

/// Knobs for the labeling pass. `spans_per_sequence` is the main
/// cost/fidelity trade-off: more samples = better per-token coverage and
/// less variance, at one extra forward pass per (sequence, span) sample.
/// Defaults are starting points, not tuned truths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    pub spans_per_sequence: usize,
    /// Span lengths are sampled uniformly in `min_span_len..=max_span_len`.
    /// OPEN QUESTION per plan: the right length distribution is untuned.
    pub min_span_len: usize,
    pub max_span_len: usize,
    pub metric: AblationMetric,
    /// Token id used to ablate (the tokenizer's MASK).
    pub mask_id: TokenId,
    pub seed: u64,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            spans_per_sequence: 16,
            min_span_len: 1,
            max_span_len: 3,
            metric: AblationMetric::Downstream,
            mask_id: crate::vocab::MASK,
            seed: 0,
        }
    }
}

/// One span's measured weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanWeight {
    pub span: Span,
    pub weight: f32,
}

/// A labeled sequence: sampled span weights plus their per-token
/// aggregation (mean weight of covering spans; `covered[p]` is false when
/// no sampled span touched position `p`, in which case `token_weights[p]`
/// is 0 by construction and should be excluded from supervised losses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledSequence {
    pub tokens: Vec<TokenId>,
    pub span_weights: Vec<SpanWeight>,
    pub token_weights: Vec<f32>,
    pub covered: Vec<bool>,
}

/// Batched scorer: for each input sequence returns per-prediction-index
/// NLL of the corresponding target row (`targets[i] = original tokens
/// shifted`, regardless of input corruption). Row length = seq_len - 1.
pub trait NllScorer {
    /// `inputs` are (possibly masked) full sequences; `originals` the
    /// uncorrupted sequence whose continuation is being scored.
    fn nll_rows(&self, inputs: &[Vec<TokenId>], original: &[TokenId]) -> Vec<Vec<f32>>;
}

impl<F> NllScorer for F
where
    F: Fn(&[Vec<TokenId>], &[TokenId]) -> Vec<Vec<f32>>,
{
    fn nll_rows(&self, inputs: &[Vec<TokenId>], original: &[TokenId]) -> Vec<Vec<f32>> {
        self(inputs, original)
    }
}

/// Sample random spans within `[1, len-1)` (BOS excluded — masking BOS is
/// uninformative; final token excluded because masking it perturbs no
/// prediction under either metric).
pub fn sample_spans(len: usize, config: &AblationConfig, rng: &mut StdRng) -> Vec<Span> {
    let lo = 1usize;
    let hi = len - 1;
    (0..config.spans_per_sequence)
        .map(|_| {
            let span_len = rng.random_range(config.min_span_len..=config.max_span_len);
            let span_len = span_len.min(hi - lo);
            let start = rng.random_range(lo..=hi - span_len);
            Span::new(start, start + span_len)
        })
        .collect()
}

/// Weight of one span given baseline and ablated NLL rows.
pub fn span_weight_from_rows(
    baseline: &[f32],
    ablated: &[f32],
    span: Span,
    metric: AblationMetric,
) -> f32 {
    let first_idx = match metric {
        AblationMetric::Downstream => span.end.saturating_sub(1),
        AblationMetric::FromSpanStart => span.start,
    };
    let deltas: Vec<f32> = (first_idx..baseline.len())
        .map(|i| ablated[i] - baseline[i])
        .collect();
    if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().sum::<f32>() / deltas.len() as f32
    }
}

/// Label one sequence: sample spans (plus any `extra_spans` the caller
/// wants measured, e.g. task-structure spans for sanity checks), score
/// baseline + all ablated variants in one batch, aggregate to tokens.
pub fn label_sequence<S: NllScorer>(
    tokens: &[TokenId],
    extra_spans: &[Span],
    scorer: &S,
    config: &AblationConfig,
    rng: &mut StdRng,
) -> LabeledSequence {
    let mut spans = sample_spans(tokens.len(), config, rng);
    spans.extend_from_slice(extra_spans);

    // One batch: row 0 = clean baseline, rows 1.. = one masked variant per span.
    let mut inputs: Vec<Vec<TokenId>> = Vec::with_capacity(spans.len() + 1);
    inputs.push(tokens.to_vec());
    for span in &spans {
        let mut masked = tokens.to_vec();
        for p in span.positions() {
            masked[p] = config.mask_id;
        }
        inputs.push(masked);
    }
    let rows = scorer.nll_rows(&inputs, tokens);
    let (baseline, ablated_rows) = rows.split_first().expect("scorer returned no rows");

    let span_weights: Vec<SpanWeight> = spans
        .iter()
        .zip(ablated_rows)
        .map(|(&span, row)| SpanWeight {
            span,
            weight: span_weight_from_rows(baseline, row, span, config.metric),
        })
        .collect();

    // Per-token aggregation: mean over covering spans. (Max is a plausible
    // alternative when long dull spans dilute short important ones —
    // untested, revisit if Critic targets look washed out.)
    let mut token_weights = vec![0.0f32; tokens.len()];
    let mut cover_counts = vec![0usize; tokens.len()];
    for sw in &span_weights {
        for p in sw.span.positions() {
            token_weights[p] += sw.weight;
            cover_counts[p] += 1;
        }
    }
    let covered: Vec<bool> = cover_counts.iter().map(|&c| c > 0).collect();
    for (w, &c) in token_weights.iter_mut().zip(&cover_counts) {
        if c > 0 {
            *w /= c as f32;
        }
    }

    LabeledSequence {
        tokens: tokens.to_vec(),
        span_weights,
        token_weights,
        covered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Fake scorer: NLL is 1.0 everywhere, except that if position 2 of the
    /// input is masked, every later prediction costs 3.0 — i.e. token 2 is
    /// the only causally-important one.
    fn fake_scorer(inputs: &[Vec<TokenId>], original: &[TokenId]) -> Vec<Vec<f32>> {
        let t = original.len();
        inputs
            .iter()
            .map(|inp| {
                let broken = inp[2] == 99;
                (0..t - 1)
                    .map(|i| if broken && i >= 2 { 3.0 } else { 1.0 })
                    .collect()
            })
            .collect()
    }

    fn test_config() -> AblationConfig {
        AblationConfig {
            spans_per_sequence: 32,
            min_span_len: 1,
            max_span_len: 2,
            metric: AblationMetric::Downstream,
            mask_id: 99,
            seed: 0,
        }
    }

    #[test]
    fn important_token_gets_higher_weight() {
        let tokens: Vec<TokenId> = vec![1, 10, 11, 12, 13, 14, 2];
        let config = test_config();
        let mut rng = StdRng::seed_from_u64(7);
        let labeled = label_sequence(&tokens, &[], &fake_scorer, &config, &mut rng);

        assert!(labeled.covered[2], "position 2 should be covered by samples");
        let w2 = labeled.token_weights[2];
        // Every other content position should have strictly lower weight.
        for p in [1usize, 3, 4, 5] {
            assert!(
                w2 > labeled.token_weights[p] + 0.5,
                "expected pos 2 ({w2}) >> pos {p} ({})",
                labeled.token_weights[p]
            );
        }
    }

    #[test]
    fn extra_spans_are_measured() {
        let tokens: Vec<TokenId> = vec![1, 10, 11, 12, 2];
        let config = AblationConfig {
            spans_per_sequence: 0,
            ..test_config()
        };
        let mut rng = StdRng::seed_from_u64(0);
        let labeled = label_sequence(
            &tokens,
            &[Span::new(2, 3), Span::new(3, 4)],
            &fake_scorer,
            &config,
            &mut rng,
        );
        assert_eq!(labeled.span_weights.len(), 2);
        assert!(labeled.span_weights[0].weight > 1.5); // masks important pos 2
        assert!(labeled.span_weights[1].weight < 0.5);
    }

    #[test]
    fn downstream_metric_ignores_in_span_targets() {
        let baseline = vec![1.0; 6];
        let ablated = vec![1.0, 5.0, 5.0, 1.0, 1.0, 1.0]; // deltas at i=1,2 only
        // Span [1,3): Downstream starts at i = e-1 = 2 → sees delta at 2 only.
        let w_down =
            span_weight_from_rows(&baseline, &ablated, Span::new(1, 3), AblationMetric::Downstream);
        let w_from = span_weight_from_rows(
            &baseline,
            &ablated,
            Span::new(1, 3),
            AblationMetric::FromSpanStart,
        );
        assert!(w_from > w_down);
    }
}
