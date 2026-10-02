//! Evaluation metrics for beat/onset timelines.
//!
//! Definitions follow mir_eval (Raffel et al., ISMIR 2014), whose reference
//! implementation is MIT-licensed; this is an independent implementation.
//! Differential testing against Python `mir_eval` is dev-only tooling
//! (never a runtime dependency).
//!
//! Matching: for sorted 1-D events with a uniform tolerance window, matching
//! each reference (in ascending order) to the *earliest* still-unmatched
//! estimate inside its window yields a maximum-cardinality matching (classic
//! exchange argument for interval bigraphs). This equals the cardinality of
//! mir_eval's Hungarian-based `util.match_events`, so F-measures agree
//! exactly; the *pairing* can differ, which only affects offset statistics.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FMeasure {
    pub precision: f64,
    pub recall: f64,
    pub f: f64,
}

/// Match reference and estimated events within ±`tolerance` seconds.
/// Returns `(reference_time, estimate_time)` pairs, sorted by reference time.
///
/// Greedy "earliest available estimate wins" — maximum-cardinality for this
/// problem class (see module docs).
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
        // Earliest unused estimate inside the window.
        let mut j = lo;
        while j < ests.len() && ests[j] <= r + tolerance {
            if !used[j] {
                used[j] = true;
                pairs.push((r, ests[j]));
                break;
            }
            j += 1;
        }
    }
    pairs
}

/// Beat F-measure with an explicit tolerance window (mir_eval default: 0.07 s).
///
/// Edge convention matches mir_eval exactly: if either sequence is empty,
/// the score is 0.0.
pub fn f_measure(reference: &[f64], estimate: &[f64], tolerance: f64) -> FMeasure {
    if reference.is_empty() || estimate.is_empty() {
        return FMeasure {
            precision: 0.0,
            recall: 0.0,
            f: 0.0,
        };
    }
    let matches = match_events(reference, estimate, tolerance).len() as f64;
    let precision = matches / estimate.len() as f64;
    let recall = matches / reference.len() as f64;
    let f = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    FMeasure {
        precision,
        recall,
        f,
    }
}

/// Signed offsets (estimate − reference) of matched events, in seconds.
/// The mean of these values exposes systematic early/late bias — which the
/// F-measure alone cannot see — and which matters for frame-accurate video
/// work. Unmatched events are excluded.
pub fn matched_offsets(reference: &[f64], estimate: &[f64], tolerance: f64) -> Vec<f64> {
    match_events(reference, estimate, tolerance)
        .iter()
        .map(|(r, e)| e - r)
        .collect()
}

/// Continuity metrics (Davies, Degara & Plumbley 2009), ported from
/// mir_eval's `beat.continuity` with identical thresholds and edge-case
/// semantics (phase and period tolerance 17.5% of the local
/// inter-annotation interval). AML* variants take the max over metrical
/// variations of the reference: as-is, offbeat, double tempo, half tempo
/// (even/odd phase).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Continuity {
    /// Correct metric level, continuous accuracy (longest correct run / len).
    pub cmlc: f64,
    /// Correct metric level, total accuracy.
    pub cmlt: f64,
    /// Any metric level, continuous accuracy.
    pub amlc: f64,
    /// Any metric level, total accuracy.
    pub amlt: f64,
}

/// mir_eval `_get_reference_beat_variations`: (as-is, offbeat, double,
/// half-even, half-odd).
fn beat_variations(reference: &[f64]) -> [Vec<f64>; 5] {
    let n = reference.len();
    let mut double = Vec::with_capacity(2 * n - 1);
    for i in 0..n {
        double.push(reference[i]);
        if i + 1 < n {
            double.push(f64::midpoint(reference[i], reference[i + 1]));
        }
    }
    [
        reference.to_vec(),
        double[1..].iter().step_by(2).copied().collect(),
        double,
        reference.iter().step_by(2).copied().collect(),
        reference[1..].iter().step_by(2).copied().collect(),
    ]
}

/// One continuity pass against a single reference variation.
/// Returns (continuous_accuracy, total_accuracy), both normalized by
/// `max(reference.len(), estimate.len())` — mir_eval's convention (it sizes
/// the success array by the larger of the two), NOT by the estimate length.
fn continuity_once(reference: &[f64], est: &[f64], phase_thr: f64, period_thr: f64) -> (f64, f64) {
    let mut used = vec![false; reference.len().max(est.len())];
    let mut success = vec![false; est.len()];

    for m in 0..est.len() {
        // Nearest reference beat; first index wins ties (np.argmin semantics).
        let mut nearest = 0;
        let mut min_diff = f64::INFINITY;
        for (i, &r) in reference.iter().enumerate() {
            let d = (est[m] - r).abs();
            if d < min_diff {
                min_diff = d;
                nearest = i;
            }
        }
        if used[nearest] {
            continue;
        }
        if m == 0 || nearest == 0 {
            // First beat or first reference: look *forward* for intervals.
            let ref_interval = if nearest + 1 < reference.len() {
                reference[nearest + 1] - reference[nearest]
            } else if reference.len() >= 2 {
                reference[nearest] - reference[nearest - 1]
            } else {
                0.0 // Python negative-index wrap: v[0] - v[-1] == 0
            };
            let phase = if ref_interval == 0.0 {
                if min_diff == 0.0 { 1.0 } else { f64::INFINITY }
            } else {
                (min_diff / ref_interval).abs()
            };
            let est_interval = if m + 1 < est.len() {
                est[m + 1] - est[m]
            } else {
                est[m] - est[m - 1]
            };
            let period = if ref_interval == 0.0 {
                if est_interval == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                (1.0 - est_interval / ref_interval).abs()
            };
            if phase < phase_thr && period < period_thr {
                used[nearest] = true;
                success[m] = true;
            }
        } else {
            let ref_interval = reference[nearest] - reference[nearest - 1];
            let phase = (min_diff / ref_interval).abs();
            let est_interval = est[m] - est[m - 1];
            let period = (1.0 - est_interval / ref_interval).abs();
            if phase < phase_thr && period < period_thr {
                used[nearest] = true;
                success[m] = true;
            }
        }
    }

    let mut longest = 0usize;
    let mut run = 0usize;
    let mut total = 0usize;
    for &s in &success {
        if s {
            run += 1;
            total += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    let denom = reference.len().max(est.len()) as f64;
    (longest as f64 / denom, total as f64 / denom)
}

pub fn continuity(reference: &[f64], estimate: &[f64]) -> Continuity {
    if reference.len() <= 1 || estimate.len() <= 1 {
        return Continuity {
            cmlc: 0.0,
            cmlt: 0.0,
            amlc: 0.0,
            amlt: 0.0,
        };
    }
    let variations = beat_variations(reference);
    let mut first = (0.0, 0.0);
    let mut amlc = 0.0f64;
    let mut amlt = 0.0f64;
    for (i, variation) in variations.iter().enumerate() {
        let (c, t) = continuity_once(variation, estimate, 0.175, 0.175);
        if i == 0 {
            first = (c, t);
        }
        amlc = amlc.max(c);
        amlt = amlt.max(t);
    }
    Continuity {
        cmlc: first.0,
        cmlt: first.1,
        amlc,
        amlt,
    }
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

    /// Regression for matching optimality: nearest-greedy would match
    /// ref 0.0 → est 0.01 and leave ref 0.06 unmatched; earliest-greedy
    /// achieves the maximum 2 matches. (mir_eval's Hungarian agrees.)
    #[test]
    fn matching_is_maximum_cardinality() {
        let m = f_measure(&[0.0, 0.06], &[0.01, -0.02], 0.05);
        assert_eq!(m.f, 1.0, "expected both references matched: {m:?}");
    }

    #[test]
    fn continuity_perfect_is_one() {
        let r = [1.0, 1.5, 2.0, 2.5, 3.0];
        let c = continuity(&r, &r);
        assert_eq!(
            c,
            Continuity {
                cmlc: 1.0,
                cmlt: 1.0,
                amlc: 1.0,
                amlt: 1.0
            }
        );
    }

    /// Double-tempo estimate: wrong period at the correct metric level (CMLt
    /// collapses), but the double-tempo reference variation rescues it (AMLt
    /// stays high) — the canonical reason AML metrics exist.
    #[test]
    fn continuity_double_tempo() {
        let reference: Vec<f64> = (0..16).map(|i| 1.0 + 0.5 * i as f64).collect();
        let estimate: Vec<f64> = (0..31).map(|i| 1.0 + 0.25 * i as f64).collect();
        let c = continuity(&reference, &estimate);
        assert!(c.cmlt < 0.2, "cmlt {}", c.cmlt);
        assert!(c.amlt > 0.8, "amlt {}", c.amlt);
    }

    #[test]
    fn continuity_empty_is_zero() {
        assert_eq!(continuity(&[], &[1.0]).cmlt, 0.0);
        assert_eq!(continuity(&[1.0], &[]).amlt, 0.0);
    }

    #[test]
    fn empty_conventions_match_mir_eval() {
        // mir_eval returns 0.0 whenever either sequence is empty.
        assert_eq!(f_measure(&[], &[], 0.07).f, 0.0);
        assert_eq!(f_measure(&[1.0], &[], 0.07).f, 0.0);
        assert_eq!(f_measure(&[], &[1.0], 0.07).f, 0.0);
    }
}
