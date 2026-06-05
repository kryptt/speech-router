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
    #[error("pcm_s16le encoder not available in this ffmpeg build")]
    EncoderNotFound,
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

/// Allocate an output frame in the resampler's target format (16 kHz mono
/// s16le) with room for `capacity_samples`.
///
/// Pre-allocating with explicit parameters is required: handed a bare
/// `AudioFrame::empty()`, `swr_convert_frame` raises `AVERROR_OUTPUT_CHANGED`
/// because the frame it auto-allocates doesn't satisfy its output-param check.
fn mono_s16_frame(capacity_samples: usize) -> AudioFrame {
    AudioFrame::new(
        ffmpeg_next::format::Sample::I16(ffmpeg_next::format::sample::Type::Packed),
        capacity_samples.max(1),
        ffmpeg_next::ChannelLayout::MONO,
    )
}

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

    // Resolve a concrete input channel layout. Raw PCM (and some other inputs)
    // report an UNSPEC layout in libav 7 — `channel_layout()` comes back empty
    // (bits = 0) even though the channel count is known. `swr_convert_frame`
    // refuses an unspecified layout and returns AVERROR_INPUT_CHANGED, which
    // failed every extraction. Fall back to the canonical default layout for
    // the channel count, and stamp the same layout onto each decoded frame
    // below so the frame matches the resampler's configured input.
    let in_layout = {
        let cl = decoder.channel_layout();
        if cl.is_empty() {
            ffmpeg_next::ChannelLayout::default(i32::from(decoder.channels()))
        } else {
            cl
        }
    };

    // Set up resampler: input format -> 16 kHz mono s16.
    let mut resampler = resampling::Context::get(
        decoder.format(),
        in_layout,
        decoder.rate(),
        ffmpeg_next::format::Sample::I16(ffmpeg_next::format::sample::Type::Packed),
        ffmpeg_next::ChannelLayout::MONO,
        TARGET_RATE,
    )?;

    // Set up output: WAV container with pcm_s16le.
    //
    // The encoder context MUST be created bound to the codec
    // (`new_with_codec`) and opened with it (`open_as`). A codec-less
    // `Context::new()` makes `avcodec_open2()` fail with "No codec provided to
    // avcodec_open2()" — which silently broke EVERY segment extraction and thus
    // all multi-chunk language detection (`/asr` returned "language detection
    // failed" / 0-of-N chunks). See the `extract_segment_produces_valid_wav`
    // regression test.
    let mut output_ctx = format::output(output_path)?;

    let pcm_codec =
        codec::encoder::find(codec::Id::PCM_S16LE).ok_or(FfmpegError::EncoderNotFound)?;

    let mut enc = codec::context::Context::new_with_codec(pcm_codec)
        .encoder()
        .audio()?;
    enc.set_rate(TARGET_RATE as i32);
    enc.set_channel_layout(ffmpeg_next::ChannelLayout::MONO);
    enc.set_format(ffmpeg_next::format::Sample::I16(
        ffmpeg_next::format::sample::Type::Packed,
    ));
    enc.set_time_base(ffmpeg_next::Rational::new(1, TARGET_RATE as i32));
    let mut encoder = enc.open_as(pcm_codec)?;

    // Register the output stream from the opened encoder, then write the
    // header. Scoped so the mutable borrow of `output_ctx` is released before
    // `write_header()`.
    let out_stream_index = {
        let mut out_stream = output_ctx.add_stream(pcm_codec)?;
        out_stream.set_parameters(&encoder);
        out_stream.index()
    };
    output_ctx.write_header()?;

    let out_time_base = output_ctx.stream(out_stream_index).unwrap().time_base();

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

            // Match the frame's layout to the resampler's configured input
            // (see `in_layout` above) so UNSPEC-layout PCM frames are accepted.
            if decoded_frame.channel_layout().is_empty() {
                decoded_frame.set_channel_layout(in_layout);
            }

            let mut resampled = mono_s16_frame(decoded_frame.samples());
            resampler.run(&decoded_frame, &mut resampled)?;

            if resampled.samples() > 0 {
                resampled.set_pts(Some(output_pts));
                output_pts += resampled.samples() as i64;

                encoder.send_frame(&resampled)?;
                receive_and_write_packets(
                    &mut encoder,
                    &mut output_ctx,
                    out_stream_index,
                    out_time_base,
                )?;
            }
        }
    }

    // Flush decoder.
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        if decoded_frame.channel_layout().is_empty() {
            decoded_frame.set_channel_layout(in_layout);
        }
        let mut resampled = mono_s16_frame(decoded_frame.samples());
        resampler.run(&decoded_frame, &mut resampled)?;
        if resampled.samples() > 0 {
            resampled.set_pts(Some(output_pts));
            output_pts += resampled.samples() as i64;

            encoder.send_frame(&resampled)?;
            receive_and_write_packets(
                &mut encoder,
                &mut output_ctx,
                out_stream_index,
                out_time_base,
            )?;
        }
    }

    // Flush resampler (buffered samples).
    loop {
        let mut resampled = mono_s16_frame(TARGET_RATE as usize);
        let delay = resampler.flush(&mut resampled)?;
        if resampled.samples() > 0 {
            resampled.set_pts(Some(output_pts));
            output_pts += resampled.samples() as i64;

            encoder.send_frame(&resampled)?;
            receive_and_write_packets(
                &mut encoder,
                &mut output_ctx,
                out_stream_index,
                out_time_base,
            )?;
        }
        if delay.is_none() {
            break;
        }
    }

    // Flush encoder.
    encoder.send_eof()?;
    receive_and_write_packets(
        &mut encoder,
        &mut output_ctx,
        out_stream_index,
        out_time_base,
    )?;

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
pub(crate) fn sample_offsets(
    total_duration: f64,
    num_samples: usize,
    sample_duration: f64,
) -> Vec<f64> {
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

    /// Write a `secs`-second 16 kHz mono s16le sine-tone WAV to `path`.
    fn write_test_wav(path: &Path, secs: u32) {
        write_test_wav_with(path, secs, 1, TARGET_RATE);
    }

    /// Write a `secs`-second interleaved s16le sine-tone WAV with the given
    /// channel count and sample rate. Used to exercise the real downmix +
    /// resample path (e.g. stereo/5.1 @ 48 kHz -> mono @ 16 kHz).
    fn write_test_wav_with(path: &Path, secs: u32, channels: u16, rate: u32) {
        use std::io::Write;

        let frames = rate * secs;
        let block_align = channels * 2; // s16le
        let data_len = frames * u32::from(block_align);
        let mut buf = Vec::with_capacity(44 + data_len as usize);

        // RIFF / WAVE header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&rate.to_le_bytes());
        buf.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes()); // byte rate
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..frames {
            let t = f64::from(i) / f64::from(rate);
            let sample = (8000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16;
            for _ in 0..channels {
                buf.extend_from_slice(&sample.to_le_bytes());
            }
        }

        let mut f = std::fs::File::create(path).expect("create test wav");
        f.write_all(&buf).expect("write test wav");
    }

    /// Regression test for the codec-less-encoder bug: `transcode_audio` once
    /// built its output encoder from a bare `Context::new()` (no codec) and
    /// called `.open()`, which fails inside `avcodec_open2` with "No codec
    /// provided to avcodec_open2()". That broke EVERY `extract_segment` call, so
    /// `/asr` multi-chunk language detection always returned 0 successful chunks
    /// ("language detection failed"). This test extracts a segment and asserts
    /// the result is a valid, non-empty, decodable WAV.
    #[test]
    fn extract_segment_produces_valid_wav() {
        init();

        let input = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();
        write_test_wav(&input, 5);

        let output = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();

        extract_segment(&input, &output, 1.0, 2.0).expect("extract_segment should succeed");

        let meta = std::fs::metadata(&output).expect("output wav exists");
        assert!(meta.len() > 44, "output wav must contain PCM beyond header");

        // The extracted segment must itself be a decodable media file.
        let dur = get_duration(&output).expect("extracted segment must be decodable");
        assert!(
            dur > 0.5,
            "extracted segment duration should be > 0.5s, got {dur}"
        );
    }

    /// Companion to the above for the full-file extraction path
    /// (`extract_audio`, used by the video/`/asr` track-selection path).
    #[test]
    fn extract_audio_produces_valid_wav() {
        init();

        let input = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();
        write_test_wav(&input, 3);

        let output = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();

        extract_audio(&input, &output, None).expect("extract_audio should succeed");

        let meta = std::fs::metadata(&output).expect("output wav exists");
        assert!(meta.len() > 44, "output wav must contain PCM beyond header");
        let dur = get_duration(&output).expect("extracted audio must be decodable");
        assert!(
            dur > 1.0,
            "extracted audio duration should be > 1s, got {dur}"
        );
    }

    /// Exercises the real downmix + downsample path that the video/`/asr` route
    /// hits with broadcast audio (e.g. 5.1 @ 48 kHz eac3): a multi-channel,
    /// non-16 kHz source must be converted to 16 kHz mono without the resampler
    /// rejecting frames. Mirrors the codec found on the failing MasterChef
    /// episodes (stereo stand-in keeps the test fixture trivial to synthesize).
    #[test]
    fn extract_segment_downmixes_and_resamples() {
        init();

        let input = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();
        // Stereo, 48 kHz — different channel count AND rate from the target.
        write_test_wav_with(&input, 4, 2, 48_000);

        let output = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .unwrap()
            .into_temp_path();

        extract_segment(&input, &output, 1.0, 2.0)
            .expect("extract_segment should downmix + resample successfully");

        let dur = get_duration(&output).expect("extracted segment must be decodable");
        assert!(
            dur > 0.5,
            "resampled segment duration should be > 0.5s, got {dur}"
        );

        // The output must be 16 kHz mono.
        let probed = format::input(&output).expect("probe output");
        let astream = probed
            .streams()
            .best(MediaType::Audio)
            .expect("output has an audio stream");
        let params = astream.parameters();
        let ctx = codec::context::Context::from_parameters(params).unwrap();
        let dec = ctx.decoder().audio().unwrap();
        assert_eq!(dec.rate(), TARGET_RATE, "output must be 16 kHz");
        assert_eq!(dec.channels(), 1, "output must be mono");
    }
}
