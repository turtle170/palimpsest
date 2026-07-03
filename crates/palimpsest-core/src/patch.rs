use serde::{Deserialize, Serialize};

use crate::span::Span;
use crate::tokenizer::TokenId;

/// A proposed replacement of one span's tokens.
///
/// v1 restriction: patches are same-length (replacement.len() == span.len()),
/// which keeps every downstream position index stable across an edit.
/// Variable-length patches are an open design question (they invalidate
/// cached Router/Critic scores positionally) — revisit after Phase 6.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub span: Span,
    pub replacement: Vec<TokenId>,
}

impl Patch {
    pub fn new(span: Span, replacement: Vec<TokenId>) -> Self {
        Self { span, replacement }
    }

    /// Whether applying this patch would change nothing (no-op edit).
    pub fn is_noop_on(&self, tokens: &[TokenId]) -> bool {
        tokens.get(self.span.start..self.span.end) == Some(self.replacement.as_slice())
    }

    /// Apply the patch to a token buffer, returning the patched copy.
    ///
    /// Panics if the span is out of bounds. Same-length patches keep indices
    /// stable; if replacement length differs the buffer is spliced (allowed,
    /// but see the struct-level note).
    pub fn apply(&self, tokens: &[TokenId]) -> Vec<TokenId> {
        assert!(
            self.span.end <= tokens.len(),
            "patch span {:?} out of bounds for sequence of length {}",
            self.span,
            tokens.len()
        );
        let mut out = Vec::with_capacity(tokens.len() - self.span.len() + self.replacement.len());
        out.extend_from_slice(&tokens[..self.span.start]);
        out.extend_from_slice(&self.replacement);
        out.extend_from_slice(&tokens[self.span.end..]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_same_length() {
        let patch = Patch::new(Span::new(1, 3), vec![9, 9]);
        assert_eq!(patch.apply(&[0, 1, 2, 3]), vec![0, 9, 9, 3]);
    }

    #[test]
    fn noop_detection() {
        let patch = Patch::new(Span::new(1, 3), vec![1, 2]);
        assert!(patch.is_noop_on(&[0, 1, 2, 3]));
        assert!(!patch.is_noop_on(&[0, 5, 2, 3]));
    }
}
