//! Audio decoding via Symphonia (pure Rust).
//!
//! Guarantees this module makes:
//! - Output is mono f32, downmixed by arithmetic mean across channels.
//! - Gapless trimming stays enabled (Symphonia default): for MP3 files with a
//!   Xing/LAME tag the encoder delay/padding is removed, so sample 0 of the
//!   output lines up with sample 0 of the original PCM. Files without the tag
//!   cannot be trimmed — see crate-level docs for the accuracy impact.
//! - `duration_seconds` is derived from the number of samples actually
//!   decoded, not container metadata.

use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::BeatlocError;

/// Decoded audio: mono f32 samples at the source's native sample rate.
pub struct DecodedAudio {
    /// Mono f32 samples (arithmetic-mean downmix).
    pub samples: Vec<f32>,
    /// Native sample rate of `samples`.
    pub sample_rate: u32,
    /// Number of channels in the source before downmixing.
    pub channels: usize,
    /// Short codec identifier, e.g. "mp3", "flac", "pcm-s16le".
    pub codec: String,
}

impl DecodedAudio {
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

pub fn decode_file(path: &Path) -> Result<DecodedAudio, BeatlocError> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut reader = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| BeatlocError::Unsupported(format!("{}: {e}", path.display())))?;

    // Copy out what we need so the track borrow ends before we iterate packets.
    let (track_id, params) = {
        let track = reader
            .default_track(TrackType::Audio)
            .ok_or_else(|| BeatlocError::Unsupported("no audio track found".into()))?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio().cloned())
            .ok_or_else(|| BeatlocError::Unsupported("audio track has no codec parameters".into()))?;
        (track.id, params)
    };

    // Gapless trimming (delay/padding, e.g. MP3 encoder delay) is on by default.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| BeatlocError::Unsupported(format!("codec not supported: {e}")))?;

    let codec = decoder.codec_info().short_name.to_string();

    let mut interleaved: Vec<f32> = Vec::new();
    // NB: `copy_to_vec_interleaved` resizes rather than appends, so each
    // packet goes through a scratch buffer.
    let mut scratch: Vec<f32> = Vec::new();
    let mut channels: Option<usize> = params.channels.map(|c| c.count());
    let mut sample_rate: Option<u32> = params.sample_rate;

    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break, // end of stream
            Err(e) => return Err(BeatlocError::Decode(e.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = buf.spec();
                channels = Some(spec.channels().count());
                sample_rate = Some(spec.rate());
                // Converts any source sample format to interleaved f32.
                buf.copy_to_vec_interleaved(&mut scratch);
                interleaved.extend_from_slice(&scratch);
            }
            // Tolerate individual corrupt packets; abort on anything else.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(BeatlocError::Decode(e.to_string())),
        }
    }

    let channels = channels.ok_or_else(|| BeatlocError::Decode("no audio frames decoded".into()))?;
    let sample_rate =
        sample_rate.ok_or_else(|| BeatlocError::Decode("sample rate unknown".into()))?;

    // Downmix to mono: arithmetic mean across channels (see crate-level docs).
    let frames = interleaved.len() / channels.max(1);
    let mut mono = vec![0.0f32; frames];
    if channels > 0 {
        for (frame, out) in mono.iter_mut().enumerate() {
            let base = frame * channels;
            let mut acc = 0.0f32;
            for c in 0..channels {
                acc += interleaved[base + c];
            }
            *out = acc / channels as f32;
        }
    }

    Ok(DecodedAudio { samples: mono, sample_rate, channels, codec })
}
