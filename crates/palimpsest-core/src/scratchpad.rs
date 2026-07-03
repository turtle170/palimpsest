use serde::{Deserialize, Serialize};

use crate::patch::Patch;
use crate::tokenizer::TokenId;

/// One applied edit in a scratchpad's history, with the signals that led
/// to it — kept so full trajectories can be dumped and inspected (Phase 6
/// explicitly requires this for debugging oscillation/no-op loops).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditRecord {
    pub step: usize,
    pub patch: Patch,
    /// Router score that flagged this span (mean over span positions).
    pub router_score: f32,
    /// Critic-predicted weight of the span before the edit.
    pub critic_weight_before: f32,
}

/// The private scratchpad: current token buffer plus full edit history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadState {
    pub tokens: Vec<TokenId>,
    /// The initial draft, retained for before/after comparison.
    pub initial_draft: Vec<TokenId>,
    pub history: Vec<EditRecord>,
}

impl ScratchpadState {
    pub fn new(draft: Vec<TokenId>) -> Self {
        Self {
            tokens: draft.clone(),
            initial_draft: draft,
            history: Vec::new(),
        }
    }

    pub fn edit_count(&self) -> usize {
        self.history.len()
    }

    /// Apply a patch and record it.
    pub fn apply(&mut self, patch: Patch, router_score: f32, critic_weight_before: f32) {
        self.tokens = patch.apply(&self.tokens);
        self.history.push(EditRecord {
            step: self.history.len(),
            patch,
            router_score,
            critic_weight_before,
        });
    }
}
