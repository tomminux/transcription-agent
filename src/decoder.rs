//! Audio decoding: any supported file -> 16 kHz mono f32 samples for Whisper.

use std::fs::File;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Sample rate required by Whisper.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decoded audio ready for transcription.
#[derive(Debug)]
pub struct DecodedAudio {
    /// Mono samples at [`WHISPER_SAMPLE_RATE`], in [-1, 1].
    pub samples: Vec<f32>,
}

impl DecodedAudio {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / WHISPER_SAMPLE_RATE as f64
    }
}

/// Decodes `path` to 16 kHz mono f32.
///
/// Uses the built-in pure-Rust decoder first. Files it cannot handle (e.g.
/// HE-AAC, more than 2 channels, unusual AAC variants) are retried with
/// `ffmpeg` if it is installed (it is in the container image).
pub fn decode_file(path: &Path) -> Result<DecodedAudio> {
    match decode_native(path) {
        Ok(audio) => Ok(audio),
        Err(native_err) => {
            tracing::warn!(
                file = %path.display(),
                "built-in decoder failed ({native_err:#}), trying ffmpeg"
            );
            match decode_ffmpeg(path) {
                Ok(samples) if !samples.is_empty() => Ok(DecodedAudio { samples }),
                Ok(_) => Err(native_err.context("ffmpeg fallback produced no audio")),
                Err(ffmpeg_err) => {
                    Err(native_err.context(format!("ffmpeg fallback failed: {ffmpeg_err:#}")))
                }
            }
        }
    }
}

fn decode_native(path: &Path) -> Result<DecodedAudio> {
    let (mono, rate) = decode_to_mono(path)?;
    if mono.is_empty() {
        bail!("no audio samples decoded from {}", path.display());
    }
    let samples = resample_mono(&mono, rate, WHISPER_SAMPLE_RATE)?;
    Ok(DecodedAudio { samples })
}

/// Decodes with the `ffmpeg` binary straight to 16 kHz mono f32 (raw, via a pipe).
fn decode_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        // "file:" keeps a path starting with "-" from being read as an option.
        .arg(format!("file:{}", path.display()))
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        .output()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => anyhow!("ffmpeg is not installed"),
            _ => anyhow!("cannot run ffmpeg: {e}"),
        })?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("ffmpeg exited with {}: {}", output.status, err.trim());
    }
    let (frames, _rest) = output.stdout.as_chunks::<4>();
    Ok(frames.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

/// Averages interleaved multi-channel samples into mono.
fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
    } else {
        out.extend(
            interleaved
                .chunks_exact(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32),
        );
    }
}

fn decode_to_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("unrecognized audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track found"))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("audio track has no codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .context("unsupported audio codec")?;

    let mut mono = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut rate = 0u32;

    while let Some(packet) = format.next_packet().context("error reading packet")? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = buf.spec();
                rate = spec.rate();
                let channels = spec.channels().count();
                interleaved.resize(buf.samples_interleaved(), f32::MID);
                buf.copy_to_slice_interleaved(&mut interleaved);
                downmix(&interleaved, channels, &mut mono);
            }
            // A corrupt packet is skipped; the rest of the file is still useful.
            Err(SymError::DecodeError(e)) => tracing::warn!("skipping undecodable packet: {e}"),
            Err(e) => return Err(e).context("decode error"),
        }
    }

    if rate == 0 {
        bail!("could not determine sample rate");
    }
    Ok((mono, rate))
}

fn resample_mono(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(input.to_vec());
    }
    let mut resampler = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)
        .context("cannot create resampler")?;
    let adapter =
        InterleavedSlice::new(input, 1, input.len()).context("invalid resampler input")?;
    let out = resampler
        .process_all(&adapter, input.len(), None)
        .context("resampling failed")?;
    Ok(out.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a 16-bit PCM WAV with a sine tone.
    fn write_wav(path: &Path, rate: u32, channels: u16, secs: f32, freq: f32) {
        let frames = (rate as f32 * secs) as usize;
        let mut data = Vec::new();
        for i in 0..frames {
            let v = (0.5
                * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin()
                * i16::MAX as f32) as i16;
            for _ in 0..channels {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn decodes_and_resamples_stereo_48k() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("t.wav");
        write_wav(&f, 48_000, 2, 2.0, 440.0);
        let a = decode_file(&f).unwrap();
        assert!(
            (a.duration_secs() - 2.0).abs() < 0.1,
            "{}",
            a.duration_secs()
        );
        let peak = a.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!((0.3..0.7).contains(&peak), "peak {peak}");
    }

    #[test]
    fn passes_through_16k_mono() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("t.wav");
        write_wav(&f, 16_000, 1, 1.0, 300.0);
        let a = decode_file(&f).unwrap();
        assert_eq!(a.samples.len(), 16_000);
    }

    #[test]
    fn rejects_garbage() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("bad.m4a");
        std::fs::write(&f, b"not audio at all").unwrap();
        assert!(decode_file(&f).is_err());
    }

    #[test]
    fn downmix_averages_channels() {
        let mut out = Vec::new();
        downmix(&[1.0, 0.0, 0.5, 0.5], 2, &mut out);
        assert_eq!(out, [0.5, 0.5]);
    }
}
