//! Synthetic toy data for validating the Palimpsest mechanism cheaply.
//!
//! The toy task is key-value recall: a sequence lists key→value pairs, then
//! queries one key; the model must emit the bound value. This was chosen
//! (over e.g. plain char-level text) because it has *known causal
//! structure*: ablating the queried pair must hurt the answer prediction,
//! ablating other pairs must not — exactly the ground truth the ablation
//! pipeline (Phase 2) and Critic (Phase 3) need to be sanity-checked
//! against.

pub mod ablation;
pub mod batch;
pub mod kv_task;
pub mod vocab;

pub use ablation::{AblationConfig, AblationMetric, LabeledSequence, SpanWeight};
pub use batch::TokenBatch;
pub use kv_task::{KvExample, KvTaskConfig, generate_examples};
pub use vocab::KvVocab;
