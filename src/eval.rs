//! Evaluation metrics for beat/onset timelines.
//!
//! Definitions follow mir_eval (Raffel et al., ISMIR 2014), whose reference
//! implementation is MIT-licensed; this is an independent implementation.
//! Differential testing against Python `mir_eval` is planned dev-only tooling
//! (never a runtime dependency).
//!
//! Divergence to note: matching here is *greedy nearest within window* after
//! sorting, while mir_eval's `util.match_events` uses optimal bipartite
//! matching. Results agree except in pathological many-candidates-in-window
//! arrangements; the differential tests will quantify any gap.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FMeasure {
    pub precision: f64,
    pub recall: f64,
    pub f: f64,
}

/// Match reference and estimated events within ±`tolerance` seconds.
/// Returns `(reference_time, estimate_time)` pairs, sorted by reference time.
fn match_events(reference: &[f64], estimate: &[f64], tolerance: f64) -> Vec<(f64, f64)> {
    let mut refs = reference.to_vec();
    refs.sort_by(f64::total_cmp);
    let mut ests = estimate.to_vec();
    ests.sort_by(f64::total_cmp);

    let mut used = vec![false; ests.len()];
    let mut pairs = Vec::new();
    let mut lo = 0;
    for r in refs {
        while lo < ests.len() && ests[lo] < r - tolerance {
            lo += 1;
        }
        let mut best: Option<usize> = None;
        let mut j = lo;
        while j < ests.len() && ests[j] <= r + tolerance {
            if !used[j] {
                best = Some(match best {
                    Some(b) if (ests[b] - r).abs() <= (ests[j] - r).abs() => b,
                    _ => j,
                });
            }
            j += 1;
        }
        if let Some(j) = best {
            used[j] = true;
            pairs.push((r, ests[j]));
        }
    }
    pairs
}

/// Beat F-measure with an explicit tolerance window (mir_eval default: 0.07 s).
///
/// Edge convention: both sequences empty → 1.0; exactly one empty → 0.0.
pub fn f_measure(reference: &[f64], estimate: &[f64], tolerance: f64) -> FMeasure {
    if reference.is_empty() && estimate.is_empty() {
        return FMeasure { precision: 1.0, recall: 1.0, f: 1.0 };
    }
    if reference.is_empty() || estimate.is_empty() {
        return FMeasure { precision: 0.0, recall: 0.0, f: 0.0 };
    }
    let matches = match_events(reference, estimate, tolerance).len() as f64;
    let precision = matches / estimate.len() as f64;
    let recall = matches / reference.len() as f64;
    let f = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    FMeasure { precision, recall, f }
}

/// Signed offsets (estimate − reference) of matched events, in seconds.
/// The mean of these values exposes systematic early/late bias — which the
/// F-measure alone cannot see — and which matters for frame-accurate video
/// work. Unmatched events are excluded.
pub fn matched_offsets(reference: &[f64], estimate: &[f64], tolerance: f64) -> Vec<f64> {
    match_events(reference, estimate, tolerance).iter().map(|(r, e)| e - r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_match_scores_one() {
        let r = [0.5, 1.0, 1.5];
        let m = f_measure(&r, &r, 0.07);
        assert_eq!(m.f, 1.0);
    }

    #[test]
    fn half_matched() {
        let m = f_measure(&[0.5, 1.0], &[0.5, 9.9], 0.07);
        assert!((m.precision - 0.5).abs() < 1e-12);
        assert!((m.recall - 0.5).abs() < 1e-12);
        assert!((m.f - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tolerance_boundary_is_inclusive() {
        let m = f_measure(&[1.0], &[1.07], 0.07);
        assert_eq!(m.f, 1.0);
        let m = f_measure(&[1.0], &[1.08], 0.07);
        assert_eq!(m.f, 0.0);
    }

    #[test]
    fn one_estimate_cannot_match_twice() {
        let m = f_measure(&[1.0, 1.01], &[1.005], 0.07);
        assert!((m.f - 2.0 * (0.5 * 1.0) / 1.5).abs() < 1e-12); // 1 match of 2 refs, 1 est
    }

    #[test]
    fn offsets_carry_sign() {
        let offs = matched_offsets(&[1.0, 2.0], &[1.02, 1.98], 0.07);
        assert_eq!(offs.len(), 2);
        assert!((offs[0] - 0.02).abs() < 1e-12);
        assert!((offs[1] + 0.02).abs() < 1e-12);
    }

    #[test]
    fn empty_conventions() {
        assert_eq!(f_measure(&[], &[], 0.07).f, 1.0);
        assert_eq!(f_measure(&[1.0], &[], 0.07).f, 0.0);
        assert_eq!(f_measure(&[], &[1.0], 0.07).f, 0.0);
    }
}
