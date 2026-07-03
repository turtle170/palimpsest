use serde::{Deserialize, Serialize};

/// A half-open token range `[start, end)` within a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "span start {start} must be <= end {end}");
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Positions covered by this span.
    pub fn positions(&self) -> impl Iterator<Item = usize> {
        self.start..self.end
    }
}

/// Merge adjacent/overlapping flagged positions into spans, capping span
/// length. Used by the edit loop to turn per-token Router scores into
/// editable spans.
pub fn merge_positions_into_spans(mut positions: Vec<usize>, max_span_len: usize) -> Vec<Span> {
    assert!(max_span_len > 0);
    positions.sort_unstable();
    positions.dedup();
    let mut spans = Vec::new();
    let mut iter = positions.into_iter();
    let Some(first) = iter.next() else {
        return spans;
    };
    let (mut start, mut end) = (first, first + 1);
    for p in iter {
        if p == end && end - start < max_span_len {
            end += 1;
        } else {
            spans.push(Span::new(start, end));
            start = p;
            end = p + 1;
        }
    }
    spans.push(Span::new(start, end));
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_adjacent_positions() {
        let spans = merge_positions_into_spans(vec![5, 3, 4, 9], 8);
        assert_eq!(spans, vec![Span::new(3, 6), Span::new(9, 10)]);
    }

    #[test]
    fn respects_max_span_len() {
        let spans = merge_positions_into_spans(vec![0, 1, 2, 3], 2);
        assert_eq!(spans, vec![Span::new(0, 2), Span::new(2, 4)]);
    }

    #[test]
    fn overlap_logic() {
        assert!(Span::new(0, 3).overlaps(&Span::new(2, 5)));
        assert!(!Span::new(0, 3).overlaps(&Span::new(3, 5)));
    }
}
