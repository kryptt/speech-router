use std::path::Path;

use ffmpeg_next::codec;
use ffmpeg_next::format;
use ffmpeg_next::media::Type as MediaType;
use ffmpeg_next::software::resampling;
use ffmpeg_next::util::frame::audio::Audio as AudioFrame;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub(crate) enum FfmpegError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),
    #[error("no audio stream found")]
    NoAudioStream,
    #[error("audio stream index {0} not found")]
    StreamNotFound(u32),
}

// ---------------------------------------------------------------------------
// Audio track metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioTrack {
    pub(crate) index: u32,
    pub(crate) codec: String,
    pub(crate) language: Option<String>,
    pub(crate) title: Option<String>,
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Initialize the ffmpeg library. Call once at startup before any other
/// ffmpeg operations.
pub fn init() {
    ffmpeg_next::init().expect("ffmpeg_next::init");
}

// ---------------------------------------------------------------------------
// Public API — probing
// ---------------------------------------------------------------------------

/// Return the duration of the media file in seconds.
pub(crate) fn get_duration(path: &Path) -> Result<f64, FfmpegError> {
    let ctx = format::input(path)?;
    let duration = ctx.duration() as f64 / f64::from(ffmpeg_next::ffi::AV_TIME_BASE);
    Ok(duration)
}

/// Probe the audio tracks in a media file.
pub(crate) fn probe_audio_tracks(path: &Path) -> Result<Vec<AudioTrack>, FfmpegError> {
    let ctx = format::input(path)?;
    let tracks = ctx
        .streams()
        .filter(|s| s.parameters().medium() == MediaType::Audio)
        .map(|s| {
            let meta = s.metadata();
            AudioTrack {
                index: s.index() as u32,
                codec: s.parameters().id().name().to_string(),
                language: meta.get("language").map(|v| v.to_string()),
                title: meta.get("title").map(|v| v.to_string()),
            }
        })
        .collect();
    Ok(tracks)
}

// ---------------------------------------------------------------------------
// Public API — extraction
// ---------------------------------------------------------------------------

/// Extract an audio track from `input_path` to a 16 kHz mono s16le WAV at
/// `output_path`. When `track` is `Some(n)`, the stream with that absolute
/// index is selected; otherwise the first audio stream is used.
pub(crate) fn extract_audio(
    input_path: &Path,
    output_path: &Path,
    track: Option<u32>,
) -> Result<(), FfmpegError> {
    transcode_audio(input_path, output_path, track, None, None)
}

/// Extract a segment of audio starting at `offset_secs` for `duration_secs`
/// seconds, writing a 16 kHz mono s16le WAV to `output_path`.
pub(crate) fn extract_segment(
    input_path: &Path,
    output_path: &Path,
    offset_secs: f64,
    duration_secs: f64,
) -> Result<(), FfmpegError> {
    transcode_audio(
        input_path,
        output_path,
        None,
        Some(offset_secs),
        Some(duration_secs),
    )
}

// ---------------------------------------------------------------------------
// Core transcode pipeline
// ---------------------------------------------------------------------------

/// Target parameters for all audio extraction: 16 kHz mono signed 16-bit.
const TARGET_RATE: u32 = 16_000;

fn transcode_audio(
    input_path: &Path,
    output_path: &Path,
    stream_index: Option<u32>,
    seek_secs: Option<f64>,
    duration_secs: Option<f64>,
) -> Result<(), FfmpegError> {
    let mut input_ctx = format::input(input_path)?;

    // Find the audio stream.
    let audio_stream_index = match stream_index {
        Some(idx) => {
            let stream = input_ctx
                .streams()
                .find(|s| s.index() as u32 == idx && s.parameters().medium() == MediaType::Audio);
            match stream {
                Some(s) => s.index(),
                None => return Err(FfmpegError::StreamNotFound(idx)),
            }
        }
        None => {
            let stream = input_ctx
                .streams()
                .best(MediaType::Audio)
                .ok_or(FfmpegError::NoAudioStream)?;
            stream.index()
        }
    };

    // Set up decoder from the audio stream's codec parameters.
    let stream = input_ctx.stream(audio_stream_index).unwrap();
    let time_base = stream.time_base();
    let decoder_ctx = codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = decoder_ctx.decoder().audio()?;

    // Seek if requested (seek in the stream's time base).
    if let Some(offset) = seek_secs {
        let ts = (offset * f64::from(ffmpeg_next::ffi::AV_TIME_BASE)) as i64;
        input_ctx.seek(ts, ..ts)?;
    }

    // Compute the end timestamp (in stream time base units) if a duration limit
    // is set.
    let end_pts: Option<i64> = match (seek_secs, duration_secs) {
        (Some(off), Some(dur)) => {
            let end_sec = off + dur;
            Some((end_sec / f64::from(time_base)).round() as i64)
        }
        (None, Some(dur)) => Some((dur / f64::from(time_base)).round() as i64),
        _ => None,
    };

    // Set up resampler: input format -> 16 kHz mono s16.
    let mut resampler = resampling::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        ffmpeg_next::format::Sample::I16(ffmpeg_next::format::sample::Type::Packed),
        ffmpeg_next::ChannelLayout::MONO,
        TARGET_RATE,
    )?;

    // Set up output: WAV container with pcm_s16le.
    let mut output_ctx = format::output(output_path)?;
    {
        let mut out_stream = output_ctx.add_stream(codec::encoder::find(codec::Id::PCM_S16LE))?;
        let encoder_ctx = codec::context::Context::new();
        // Configure encoder parameters for PCM s16le.
        let mut enc = encoder_ctx.encoder().audio()?;
        enc.set_rate(TARGET_RATE as i32);
        enc.set_channel_layout(ffmpeg_next::ChannelLayout::MONO);
        enc.set_format(ffmpeg_next::format::Sample::I16(
            ffmpeg_next::format::sample::Type::Packed,
        ));
        enc.set_time_base(ffmpeg_next::Rational::new(1, TARGET_RATE as i32));
        let encoder = enc.open()?;
        out_stream.set_parameters(&encoder);
    }
    output_ctx.write_header()?;

    let out_stream_index = 0usize;
    let out_time_base = output_ctx.stream(out_stream_index).unwrap().time_base();

    // Build the encoder from the output stream parameters so we can feed it
    // resampled frames.
    let out_params = output_ctx.stream(out_stream_index).unwrap().parameters();
    let enc_ctx = codec::context::Context::from_parameters(out_params)?;
    let mut encoder = enc_ctx.encoder().audio()?;
    encoder.set_rate(TARGET_RATE as i32);
    encoder.set_channel_layout(ffmpeg_next::ChannelLayout::MONO);
    encoder.set_format(ffmpeg_next::format::Sample::I16(
        ffmpeg_next::format::sample::Type::Packed,
    ));
    encoder.set_time_base(ffmpeg_next::Rational::new(1, TARGET_RATE as i32));
    let mut encoder = encoder.open()?;

    let mut output_pts: i64 = 0;

    // Process packets.
    let mut decoded_frame = AudioFrame::empty();
    for (stream, packet) in input_ctx.packets() {
        if stream.index() != audio_stream_index {
            continue;
        }

        // Check duration limit against packet PTS.
        if let Some(end) = end_pts {
            if let Some(pts) = packet.pts() {
                if pts >= end {
                    break;
                }
            }
        }

        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            // Check frame PTS against duration limit.
            if let Some(end) = end_pts {
                if let Some(pts) = decoded_frame.pts() {
                    if pts >= end {
                        break;
                    }
                }
            }

            let mut resampled = AudioFrame::empty();
            resampler.run(&decoded_frame, &mut resampled)?;

            if resampled.samples() > 0 {
                resampled.set_pts(Some(output_pts));
                output_pts += resampled.samples() as i64;

                encoder.send_frame(&resampled)?;
                receive_and_write_packets(&mut encoder, &mut output_ctx, out_stream_index, out_time_base)?;
            }
        }
    }

    // Flush decoder.
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resampled = AudioFrame::empty();
        resampler.run(&decoded_frame, &mut resampled)?;
        if resampled.samples() > 0 {
            resampled.set_pts(Some(output_pts));
            output_pts += resampled.samples() as i64;

            encoder.send_frame(&resampled)?;
            receive_and_write_packets(&mut encoder, &mut output_ctx, out_stream_index, out_time_base)?;
        }
    }

    // Flush resampler (buffered samples).
    loop {
        let mut resampled = AudioFrame::empty();
        let delay = resampler.flush(&mut resampled)?;
        if resampled.samples() > 0 {
            resampled.set_pts(Some(output_pts));
            output_pts += resampled.samples() as i64;

            encoder.send_frame(&resampled)?;
            receive_and_write_packets(&mut encoder, &mut output_ctx, out_stream_index, out_time_base)?;
        }
        if delay.is_none() {
            break;
        }
    }

    // Flush encoder.
    encoder.send_eof()?;
    receive_and_write_packets(&mut encoder, &mut output_ctx, out_stream_index, out_time_base)?;

    output_ctx.write_trailer()?;

    Ok(())
}

/// Drain encoded packets from the encoder and write them to the output.
fn receive_and_write_packets(
    encoder: &mut codec::encoder::Audio,
    output_ctx: &mut format::context::Output,
    stream_index: usize,
    out_time_base: ffmpeg_next::Rational,
) -> Result<(), FfmpegError> {
    let mut encoded = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(stream_index);
        encoded.rescale_ts(
            ffmpeg_next::Rational::new(1, TARGET_RATE as i32),
            out_time_base,
        );
        encoded.write_interleaved(output_ctx)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Calculate evenly spaced sample offsets across a media file.
///
/// Returns `num_samples` offsets distributed uniformly. If the total duration
/// is shorter than `sample_duration`, returns a single offset at 0.0.
pub(crate) fn sample_offsets(total_duration: f64, num_samples: usize, sample_duration: f64) -> Vec<f64> {
    if total_duration <= sample_duration || num_samples <= 1 {
        return vec![0.0];
    }

    let usable = total_duration - sample_duration;
    let step = usable / (num_samples - 1) as f64;

    (0..num_samples).map(|i| step * i as f64).collect()
}

// ---------------------------------------------------------------------------
// Video-file detection
// ---------------------------------------------------------------------------

/// File extensions we treat as video containers.
const VIDEO_EXTENSIONS: &[&str] = &[
    "mkv", "mp4", "avi", "ts", "m2ts", "mov", "wmv", "flv", "webm", "mpg", "mpeg",
];

/// Returns `true` when the file name suggests a video container.
pub(crate) fn is_video_file(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    VIDEO_EXTENSIONS
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_offsets_short_file() {
        let offsets = sample_offsets(20.0, 13, 30.0);
        assert_eq!(offsets, vec![0.0]);
    }

    #[test]
    fn sample_offsets_exact_one_sample() {
        let offsets = sample_offsets(120.0, 1, 30.0);
        assert_eq!(offsets, vec![0.0]);
    }

    #[test]
    fn sample_offsets_normal() {
        let offsets = sample_offsets(120.0, 13, 30.0);
        assert_eq!(offsets.len(), 13);
        assert!((offsets[0] - 0.0).abs() < 1e-9);
        assert!((offsets[12] - 90.0).abs() < 1e-9);
        for pair in offsets.windows(2) {
            assert!(pair[1] > pair[0]);
        }
    }

    #[test]
    fn sample_offsets_two_samples() {
        let offsets = sample_offsets(120.0, 2, 30.0);
        assert_eq!(offsets.len(), 2);
        assert!((offsets[0] - 0.0).abs() < 1e-9);
        assert!((offsets[1] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn is_video_detects_common_formats() {
        assert!(is_video_file("movie.mkv"));
        assert!(is_video_file("clip.MP4"));
        assert!(is_video_file("recording.avi"));
        assert!(is_video_file("broadcast.ts"));
        assert!(is_video_file("film.webm"));
    }

    #[test]
    fn is_video_rejects_audio_files() {
        assert!(!is_video_file("song.mp3"));
        assert!(!is_video_file("audio.wav"));
        assert!(!is_video_file("voice.flac"));
        assert!(!is_video_file("podcast.ogg"));
    }
}
