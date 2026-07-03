//! Shared types for the Palimpsest self-editing scratchpad architecture.
//!
//! This crate deliberately stays small and mostly tensor-free: plain data
//! types (spans, patches, scratchpad state) live here so every component
//! crate can exchange them without pulling in each other's model code.
//! The one tensor-touching module is [`tensor_util`], which holds loss
//! helpers shared by training (Phase 1) and ablation labeling (Phase 2).

pub mod nn;
pub mod patch;
pub mod scratchpad;
pub mod span;
pub mod tensor_util;
pub mod tokenizer;

pub use patch::Patch;
pub use scratchpad::{EditRecord, ScratchpadState};
pub use span::Span;
pub use tokenizer::{CharTokenizer, TokenId, Tokenizer};
