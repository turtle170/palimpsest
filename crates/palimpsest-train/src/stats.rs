//! Small statistics helpers for phase checkpoints. The Critic checkpoint
//! is a *measured* rank correlation, not an eyeball judgment — hence a
//! proper Spearman with midranks for ties (labels contain many exact
//! zeros, so tie handling matters).

/// Spearman rank correlation between two equal-length slices.
/// Returns 0.0 when either input has zero rank variance (constant input).
pub fn spearman(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.len() < 2 {
        return 0.0;
    }
    pearson(&midranks(a), &midranks(b))
}

fn midranks(x: &[f32]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).expect("NaN in rank input"));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        // Average rank for the tie group [i, j], 1-based.
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Overlap fraction |topk(a) ∩ topk(b)| / k — the Router checkpoint metric.
pub fn top_k_overlap(a: &[f32], b: &[f32], k: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    let k = k.min(a.len());
    if k == 0 {
        return 0.0;
    }
    let top = |x: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[j].partial_cmp(&x[i]).expect("NaN in top-k input"));
        idx.truncate(k);
        idx
    };
    let ta = top(a);
    let tb = top(b);
    ta.iter().filter(|i| tb.contains(i)).count() as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_monotone_correlation() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-9);
        let inv = [40.0f32, 30.0, 20.0, 10.0];
        assert!((spearman(&a, &inv) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn ties_are_midranked() {
        // Ties everywhere in b → zero variance → 0.
        let a = [1.0f32, 2.0, 3.0];
        let b = [5.0f32, 5.0, 5.0];
        assert_eq!(spearman(&a, &b), 0.0);
        // Partial ties still yield a sensible positive correlation.
        let c = [0.0f32, 0.0, 1.0, 2.0];
        let d = [0.1f32, 0.0, 3.0, 9.0];
        assert!(spearman(&c, &d) > 0.7);
    }

    #[test]
    fn overlap() {
        let a = [9.0f32, 1.0, 8.0, 0.0];
        let b = [7.0f32, 6.0, 5.0, 0.0];
        // top-2(a) = {0, 2}, top-2(b) = {0, 1} → overlap 1/2.
        assert!((top_k_overlap(&a, &b, 2) - 0.5).abs() < 1e-9);
    }
}
