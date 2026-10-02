//! Shared STFT magnitude computation — the single definition of the
//! analysis frame grid. Both the onset envelope and the section features
//! build on this, so they can never drift apart.

use realfft::RealFftPlanner;

/// Magnitude spectra over the shared frame grid (periodic Hann window,
/// valid/unpadded frames; frame `i` covers `[i*hop, i*hop + window)`).
/// Returns one vector of `window/2 + 1` magnitudes per frame; empty when the
/// signal is shorter than one window.
pub fn stft_magnitudes(samples: &[f32], window: usize, hop: usize) -> Vec<Vec<f32>> {
    if samples.len() < window || hop == 0 || window == 0 {
        return Vec::new();
    }
    let n_frames = (samples.len() - window) / hop + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window);
    let mut in_buf = fft.make_input_vec();
    let mut out_buf = fft.make_output_vec();

    let win: Vec<f32> = (0..window)
        .map(|n| 0.5 - 0.5 * (std::f32::consts::TAU * n as f32 / window as f32).cos())
        .collect();

    let mut frames = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let start = frame * hop;
        for (dst, (&s, &w)) in in_buf
            .iter_mut()
            .zip(samples[start..start + window].iter().zip(win.iter()))
        {
            *dst = s * w;
        }
        fft.process(&mut in_buf, &mut out_buf)
            .expect("buffer sizes are exact by construction");
        frames.push(out_buf.iter().map(|c| c.norm()).collect());
    }
    frames
}
