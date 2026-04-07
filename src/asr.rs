use std::collections::HashMap;
use std::path::Path;

use axum::extract::{Multipart, Query, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use futures_util::TryStreamExt;

use crate::ffmpeg::{self, AudioTrack};
use crate::proxy::error_response;
use crate::wyoming::wav_header;

// ---------------------------------------------------------------------------
// Query-param types
// ---------------------------------------------------------------------------

/// Output format requested by the whisper-asr-webservice client.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AsrOutput {
    #[default]
    Txt,
    Srt,
    Vtt,
    Json,
    Tsv,
}

/// Speaches `response_format` value corresponding to an [`AsrOutput`].
///
/// Returns `None` for formats Speaches does not support (TSV).
fn map_output_format(output: AsrOutput) -> Option<&'static str> {
    match output {
        AsrOutput::Txt => Some("text"),
        AsrOutput::Srt => Some("srt"),
        AsrOutput::Vtt => Some("vtt"),
        AsrOutput::Json => Some("verbose_json"),
        AsrOutput::Tsv => None,
    }
}

/// Content-Type header for a given Speaches response format.
fn content_type_for(output: AsrOutput) -> &'static str {
    match output {
        AsrOutput::Txt => "text/plain",
        AsrOutput::Srt => "text/plain",
        AsrOutput::Vtt => "text/vtt",
        AsrOutput::Json => "application/json",
        AsrOutput::Tsv => "text/plain", // unreachable in practice — TSV is rejected earlier
    }
}

/// Task requested by the client — determines the Speaches endpoint.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AsrTask {
    #[default]
    Transcribe,
    Translate,
}

impl AsrTask {
    /// Speaches endpoint path for this task.
    fn endpoint(self) -> &'static str {
        match self {
            AsrTask::Transcribe => "/v1/audio/transcriptions",
            AsrTask::Translate => "/v1/audio/translations",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AsrParams {
    language: Option<String>,
    #[serde(default)]
    output: AsrOutput,
    #[serde(default = "default_true")]
    encode: bool,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Language code → name mapping (via ISO 639-1, provided by `isolang`)
// ---------------------------------------------------------------------------

/// Map a two-letter ISO 639-1 language code to its English name.
/// Returns `"Unknown"` for unrecognised codes.
fn language_name(code: &str) -> &'static str {
    isolang::Language::from_639_1(code)
        .map(|lang| lang.to_name())
        .unwrap_or("Unknown")
}

// ---------------------------------------------------------------------------
// Task decision logic
// ---------------------------------------------------------------------------

/// The result of comparing the requested language against the detected language.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskDecision {
    /// Transcribe in the detected language (requested == detected).
    Transcribe { language: String },
    /// Translate to English (Whisper can only translate *to* English).
    Translate,
    /// Cannot produce subtitles in the requested language.
    Error(String),
}

/// Decide whether to transcribe, translate, or error based on the requested
/// and detected languages.
///
/// Whisper's `translate` task always targets English. Therefore:
/// - requested == detected  -> transcribe in that language
/// - requested == "en", detected != "en" -> translate (any -> en)
/// - requested != "en", detected != requested -> error (cannot translate to
///   non-English)
fn decide_task(requested_lang: &str, detected_lang: &str) -> TaskDecision {
    if requested_lang == detected_lang {
        TaskDecision::Transcribe {
            language: detected_lang.to_string(),
        }
    } else if requested_lang == "en" {
        TaskDecision::Translate
    } else {
        TaskDecision::Error(format!(
            "audio is in '{detected_lang}' but '{requested_lang}' was requested; \
             Whisper can only translate to English"
        ))
    }
}

// ---------------------------------------------------------------------------
// Audio track selection
// ---------------------------------------------------------------------------

/// Convert a 2-letter ISO 639-1 code to its 3-letter ISO 639-2/B equivalent.
///
/// Returns `None` for unrecognised codes.
fn iso639_1_to_3(code: &str) -> Option<&'static str> {
    isolang::Language::from_639_1(code).map(|lang| lang.to_639_3())
}

/// Find the first audio track whose language tag matches `lang` (ISO 639-1).
///
/// Audio tracks use ISO 639-2 (3-letter) tags while Whisper uses ISO 639-1
/// (2-letter). This function converts `lang` to 639-2 before comparison.
fn find_track_by_language<'a>(
    tracks: &'a [AudioTrack],
    lang: &str,
) -> Option<&'a AudioTrack> {
    let target_3 = iso639_1_to_3(lang)?;
    tracks.iter().find(|t| {
        t.language
            .as_deref()
            .is_some_and(|tag| tag.eq_ignore_ascii_case(target_3))
    })
}

// ---------------------------------------------------------------------------
// Shared state expected by handlers
// ---------------------------------------------------------------------------

/// The handler state — must be extractable from the app router.
#[derive(Clone)]
pub struct AsrState {
    pub speaches_url: String,
    pub default_model: String,
    pub client: std::sync::Arc<reqwest::Client>,
}

// ---------------------------------------------------------------------------
// POST /asr
// ---------------------------------------------------------------------------

pub async fn handle_asr(
    State(state): State<AsrState>,
    Query(params): Query<AsrParams>,
    multipart: Multipart,
) -> Response {
    let response_format = match map_output_format(params.output) {
        Some(fmt) => fmt,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "tsv output format is not supported",
            );
        }
    };

    let mut audio = match extract_audio_file(multipart).await {
        Ok(a) => a,
        Err(resp) => return resp,
    };

    // When encode=false the upload is raw 16-bit signed-integer PCM at 16 kHz
    // mono.  Speaches expects an audio container format it can decode, so wrap
    // the raw samples in a WAV header.
    if !params.encode {
        audio = match wrap_raw_pcm_as_wav(audio).await {
            Ok(a) => a,
            Err(resp) => return resp,
        };
    }

    let requested_lang = params
        .language
        .as_deref()
        .filter(|l| !l.is_empty())
        .unwrap_or("en");

    match select_and_transcribe(
        &audio.temp_path,
        &audio.file_name,
        requested_lang,
        params.output,
        response_format,
        &state.default_model,
        &state.speaches_url,
        &state.client,
    )
    .await
    {
        Ok(resp) => resp,
        Err(resp) => resp,
    }
}

// ---------------------------------------------------------------------------
// Core pipeline: audio track selection + transcription
// ---------------------------------------------------------------------------

/// Select the right audio track (for videos) or use the file directly (for
/// audio), detect the language, decide transcribe vs. translate, and call
/// Speaches.
///
/// Returns the final HTTP response on success, or a 500 error response.
async fn select_and_transcribe(
    file_path: &Path,
    file_name: &str,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    speaches_url: &str,
    client: &reqwest::Client,
) -> Result<Response, Response> {
    if ffmpeg::is_video_file(file_name) {
        select_and_transcribe_video(
            file_path,
            requested_lang,
            output,
            response_format,
            model,
            speaches_url,
            client,
        )
        .await
    } else {
        select_and_transcribe_audio(
            file_path,
            requested_lang,
            output,
            response_format,
            model,
            speaches_url,
            client,
        )
        .await
    }
}

/// Handle a video file: probe tracks, try the language-matched track first,
/// fall back to track 0, then give up.
async fn select_and_transcribe_video(
    video_path: &Path,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    speaches_url: &str,
    client: &reqwest::Client,
) -> Result<Response, Response> {
    let video_path_owned = video_path.to_path_buf();
    let tracks = tokio::task::spawn_blocking(move || ffmpeg::probe_audio_tracks(&video_path_owned))
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "probe task panicked");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to probe audio tracks",
            )
        })?
        .map_err(|e| {
        tracing::warn!(error = %e, "ffprobe audio track probing failed");
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to probe audio tracks",
        )
    })?;

    tracing::info!(
        track_count = tracks.len(),
        tracks = ?tracks,
        requested_lang,
        "probed audio tracks"
    );

    // Determine which tracks to try, in order.
    // 1. Track matching the requested language (if any)
    // 2. Default track (index 0 in the audio stream list)
    let matched_track = find_track_by_language(&tracks, requested_lang);

    let attempts: Vec<Option<u32>> = match matched_track {
        Some(t) => {
            // If the matched track IS the first track, only try once.
            if tracks.first().is_some_and(|first| first.index == t.index) {
                vec![Some(t.index)]
            } else {
                vec![Some(t.index), tracks.first().map(|t| t.index)]
            }
        }
        // No language match — try the default track only.
        None => vec![tracks.first().map(|t| t.index)],
    };

    for track_index in attempts {
        let audio = extract_video_audio_track(video_path, track_index).await?;

        let detected = detect_language_from_file(speaches_url, model, client, &audio.temp_path)
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "language detection failed");
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "language detection failed",
                )
            })?;

        tracing::info!(
            ?track_index,
            detected_lang = %detected,
            requested_lang,
            "detected language for track"
        );

        match decide_task(requested_lang, &detected) {
            TaskDecision::Transcribe { language } => {
                return send_to_speaches(
                    &audio.temp_path,
                    &language,
                    AsrTask::Transcribe,
                    output,
                    response_format,
                    model,
                    speaches_url,
                    client,
                )
                .await;
            }
            TaskDecision::Translate => {
                // Whisper translate always targets English; pass the detected
                // language so Whisper knows the source.
                return send_to_speaches(
                    &audio.temp_path,
                    &detected,
                    AsrTask::Translate,
                    output,
                    response_format,
                    model,
                    speaches_url,
                    client,
                )
                .await;
            }
            TaskDecision::Error(reason) => {
                tracing::info!(
                    ?track_index,
                    reason,
                    "track does not match requested language, trying next"
                );
                continue;
            }
        }
    }

    Err(error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        &format!(
            "unable to produce subtitles in '{requested_lang}': \
             no audio track matched the requested language"
        ),
    ))
}

/// Handle a plain audio file (single track): detect language, decide task.
async fn select_and_transcribe_audio(
    audio_path: &Path,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    speaches_url: &str,
    client: &reqwest::Client,
) -> Result<Response, Response> {
    let detected =
        detect_language_from_file(speaches_url, model, client, audio_path)
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "language detection failed");
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "language detection failed",
                )
            })?;

    tracing::info!(
        detected_lang = %detected,
        requested_lang,
        "detected language for audio file"
    );

    match decide_task(requested_lang, &detected) {
        TaskDecision::Transcribe { language } => {
            send_to_speaches(
                audio_path,
                &language,
                AsrTask::Transcribe,
                output,
                response_format,
                model,
                speaches_url,
                client,
            )
            .await
        }
        TaskDecision::Translate => {
            send_to_speaches(
                audio_path,
                &detected,
                AsrTask::Translate,
                output,
                response_format,
                model,
                speaches_url,
                client,
            )
            .await
        }
        TaskDecision::Error(reason) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("unable to produce subtitles in '{requested_lang}': {reason}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Speaches request builder
// ---------------------------------------------------------------------------

/// Build and send a transcription/translation request to Speaches, returning
/// the formatted HTTP response.
async fn send_to_speaches(
    audio_path: &Path,
    language: &str,
    task: AsrTask,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    speaches_url: &str,
    client: &reqwest::Client,
) -> Result<Response, Response> {
    let file = tokio::fs::File::open(audio_path).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to open audio file for speaches");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;

    let stream = tokio_util::io::ReaderStream::new(file);
    let part = reqwest::multipart::Part::stream(reqwest::Body::wrap_stream(stream))
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to build multipart part");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
        })?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", response_format.to_string())
        .text("temperature", "0.0")
        .text("language", language.to_string());

    let url = format!("{}{}", speaches_url, task.endpoint());

    let upstream = client.post(&url).multipart(form).send().await.map_err(|e| {
        tracing::warn!(error = %e, "speaches request failed");
        error_response(StatusCode::BAD_GATEWAY, "upstream unavailable")
    })?;

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let body = upstream.bytes().await.map_err(|e| {
        tracing::warn!(error = %e, "failed to read speaches response body");
        error_response(StatusCode::BAD_GATEWAY, "failed to read upstream response")
    })?;

    let content_type = content_type_for(output);
    let mut response = (status, body).into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );

    if output == AsrOutput::Srt {
        response.headers_mut().insert(
            axum::http::header::CONTENT_DISPOSITION,
            HeaderValue::from_static("attachment; filename=\"transcription.srt\""),
        );
    }

    Ok(response)
}

// ---------------------------------------------------------------------------
// POST /detect-language
// ---------------------------------------------------------------------------

pub async fn handle_detect_language(
    State(state): State<AsrState>,
    multipart: Multipart,
) -> Response {
    let audio = match extract_audio_file(multipart).await {
        Ok(a) => a,
        Err(resp) => return resp,
    };

    // If the upload is a video container, extract its first audio track.
    let audio = if ffmpeg::is_video_file(&audio.file_name) {
        match extract_video_audio_track(&audio.temp_path, None).await {
            Ok(a) => a,
            Err(resp) => return resp,
        }
    } else {
        audio
    };

    let code = match detect_language_from_file(
        &state.speaches_url,
        &state.default_model,
        &state.client,
        &audio.temp_path,
    )
    .await
    {
        Ok(lang) => lang,
        Err(e) => {
            tracing::warn!(error = %e, "multi-chunk language detection failed");
            return error_response(StatusCode::BAD_GATEWAY, "language detection failed");
        }
    };

    let name = language_name(&code);

    let result = serde_json::json!({
        "detected_language": name,
        "language_code": code,
    });

    (StatusCode::OK, axum::Json(result)).into_response()
}

// ---------------------------------------------------------------------------
// Video audio extraction
// ---------------------------------------------------------------------------

/// Extract an audio track from a video file into a 16 kHz mono WAV.
///
/// When `track` is `Some(n)`, the n-th audio stream is selected; otherwise the
/// first audio stream is used.
async fn extract_video_audio_track(
    video_path: &Path,
    track: Option<u32>,
) -> Result<AudioFile, Response> {
    let named = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to create temp file for video audio extraction");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to create temp file",
            )
        })?;
    let wav_path = named.into_temp_path();

    let vp = video_path.to_path_buf();
    let wp = wav_path.to_path_buf();
    tokio::task::spawn_blocking(move || ffmpeg::extract_audio(&vp, &wp, track))
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "extract_audio task panicked");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to extract audio from video",
            )
        })?
        .map_err(|e| {
            tracing::warn!(error = %e, ?track, "ffmpeg audio extraction failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to extract audio from video",
            )
        })?;

    Ok(AudioFile {
        temp_path: wav_path,
        file_name: "audio.wav".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Multi-chunk language detection
// ---------------------------------------------------------------------------

const DETECT_NUM_SAMPLES: usize = 13;
const DETECT_SAMPLE_DURATION: f64 = 30.0;

/// Detect the language of an audio file using multi-chunk majority voting.
///
/// This is the reusable entry point called by both the `/asr` pipeline and
/// the `/detect-language` endpoint.
async fn detect_language_from_file(
    speaches_url: &str,
    model: &str,
    client: &reqwest::Client,
    audio_path: &Path,
) -> Result<String, String> {
    let ap = audio_path.to_path_buf();
    let duration = tokio::task::spawn_blocking(move || ffmpeg::get_duration(&ap))
        .await
        .map_err(|e| format!("get_duration task panicked: {e}"))?
        .map_err(|e| format!("ffprobe duration failed: {e}"))?;

    let offsets = ffmpeg::sample_offsets(duration, DETECT_NUM_SAMPLES, DETECT_SAMPLE_DURATION);

    if offsets.len() == 1 && offsets[0] == 0.0 && duration <= DETECT_SAMPLE_DURATION {
        // Short file: send it directly without segmenting.
        return detect_language_single(speaches_url, model, client, audio_path).await;
    }

    // Extract segments and detect concurrently.
    let mut handles = Vec::with_capacity(offsets.len());

    for offset in &offsets {
        let speaches_url = speaches_url.to_string();
        let model = model.to_string();
        let client = client.clone();
        let offset = *offset;
        let audio_path = audio_path.to_path_buf();

        handles.push(tokio::spawn(async move {
            let named = tempfile::Builder::new()
                .suffix(".wav")
                .tempfile()
                .map_err(|e| format!("temp file creation failed: {e}"))?;
            let segment_path = named.into_temp_path();

            let ap = audio_path;
            let sp = segment_path.to_path_buf();
            tokio::task::spawn_blocking(move || {
                ffmpeg::extract_segment(&ap, &sp, offset, DETECT_SAMPLE_DURATION)
            })
            .await
            .map_err(|e| format!("extract_segment task panicked: {e}"))?
            .map_err(|e| format!("segment extraction at {offset:.1}s failed: {e}"))?;

            detect_language_single(&speaches_url, &model, &client, &segment_path).await
        }));
    }

    let mut votes: HashMap<String, usize> = HashMap::new();

    for handle in handles {
        match handle.await {
            Ok(Ok(lang)) => {
                *votes.entry(lang).or_insert(0) += 1;
            }
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "chunk language detection failed, skipping");
            }
            Err(e) => {
                tracing::debug!(error = %e, "chunk detection task panicked, skipping");
            }
        }
    }

    votes
        .into_iter()
        .max_by_key(|(_lang, count)| *count)
        .map(|(lang, _)| lang)
        .ok_or_else(|| "all chunk detections failed".to_string())
}

/// Detect the language of a single audio file by sending it to Speaches.
async fn detect_language_single(
    speaches_url: &str,
    model: &str,
    client: &reqwest::Client,
    audio_path: &Path,
) -> Result<String, String> {
    let file = tokio::fs::File::open(audio_path)
        .await
        .map_err(|e| format!("failed to open audio file: {e}"))?;

    let stream = tokio_util::io::ReaderStream::new(file);
    let part = reqwest::multipart::Part::stream(reqwest::Body::wrap_stream(stream))
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("failed to build multipart part: {e}"))?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", "verbose_json")
        .text("temperature", "0.0");

    let url = format!("{speaches_url}/v1/audio/transcriptions");

    let upstream = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("speaches request failed: {e}"))?;

    if !upstream.status().is_success() {
        return Err(format!("speaches returned {}", upstream.status()));
    }

    let body: serde_json::Value = upstream
        .json()
        .await
        .map_err(|e| format!("failed to parse verbose_json: {e}"))?;

    body.get("language")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "no language field in response".to_string())
}

// ---------------------------------------------------------------------------
// Multipart audio extraction  (streams to temp file, no memory limit)
// ---------------------------------------------------------------------------

struct AudioFile {
    temp_path: tempfile::TempPath, // auto-deletes on drop
    file_name: String,
}

async fn extract_audio_file(mut multipart: Multipart) -> Result<AudioFile, Response> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "multipart parse error");
            error_response(StatusCode::BAD_REQUEST, "invalid multipart body")
        })?
    {
        if field.name() == Some("audio_file") {
            let file_name = field
                .file_name()
                .unwrap_or("audio.wav")
                .to_string();
            // Write the upload to a temp file instead of buffering in memory.
            let named = tempfile::NamedTempFile::new().map_err(|e| {
                tracing::warn!(error = %e, "failed to create temp file");
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create temp file",
                )
            })?;
            let temp_path = named.into_temp_path();

            let mut out = tokio::fs::File::create(&temp_path).await.map_err(|e| {
                tracing::warn!(error = %e, "failed to open temp file for writing");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
            })?;

            // Stream chunks from the multipart field to disk.
            let mut stream = field.into_stream();
            while let Some(chunk) = stream.try_next().await.map_err(|e| {
                tracing::warn!(error = %e, "failed to read audio_file chunk");
                error_response(StatusCode::BAD_REQUEST, "failed to read audio_file")
            })? {
                out.write_all(&chunk).await.map_err(|e| {
                    tracing::warn!(error = %e, "failed to write audio chunk to temp file");
                    error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
                })?;
            }
            out.flush().await.map_err(|e| {
                tracing::warn!(error = %e, "failed to flush temp file");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
            })?;

            return Ok(AudioFile {
                temp_path,
                file_name,
            });
        }
    }

    Err(error_response(
        StatusCode::BAD_REQUEST,
        "missing audio_file field",
    ))
}

// ---------------------------------------------------------------------------
// Raw PCM → WAV wrapper
// ---------------------------------------------------------------------------

/// Wrap raw 16-bit signed-integer PCM (16 kHz, mono) in a WAV container so
/// that Speaches can decode it.  Returns a new [`AudioFile`] backed by a fresh
/// temp file.
///
/// Streams the PCM data from the source file to the output file without loading
/// the entire payload into memory (raw PCM for a movie can exceed 200 MB).
async fn wrap_raw_pcm_as_wav(audio: AudioFile) -> Result<AudioFile, Response> {
    let pcm_len = tokio::fs::metadata(&audio.temp_path)
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to stat raw PCM temp file");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
        })?
        .len();

    let header = wav_header(pcm_len as u32, 16_000, 1, 2);

    let named = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to create WAV temp file");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "failed to create temp file")
        })?;
    let wav_path = named.into_temp_path();

    let mut out = tokio::fs::File::create(&wav_path).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to open WAV temp file for writing");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;
    out.write_all(&header).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to write WAV header");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;

    let mut src = tokio::fs::File::open(&audio.temp_path).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to open raw PCM temp file");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;
    tokio::io::copy(&mut src, &mut out).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to copy PCM data");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;

    out.flush().await.map_err(|e| {
        tracing::warn!(error = %e, "failed to flush WAV temp file");
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal I/O error")
    })?;

    Ok(AudioFile {
        temp_path: wav_path,
        file_name: "audio.wav".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_mapping() {
        assert_eq!(map_output_format(AsrOutput::Txt), Some("text"));
        assert_eq!(map_output_format(AsrOutput::Srt), Some("srt"));
        assert_eq!(map_output_format(AsrOutput::Vtt), Some("vtt"));
        assert_eq!(map_output_format(AsrOutput::Json), Some("verbose_json"));
    }

    #[test]
    fn tsv_output_returns_none() {
        assert_eq!(map_output_format(AsrOutput::Tsv), None);
    }

    #[test]
    fn language_code_to_name() {
        assert_eq!(language_name("en"), "English");
        assert_eq!(language_name("es"), "Spanish");
        assert_eq!(language_name("fr"), "French");
        assert_eq!(language_name("de"), "German");
        assert_eq!(language_name("nl"), "Dutch");
        assert_eq!(language_name("ja"), "Japanese");
        assert_eq!(language_name("zh"), "Chinese");
        assert_eq!(language_name("unknown"), "Unknown");
        assert_eq!(language_name("xx"), "Unknown");
    }

    #[test]
    fn task_endpoints() {
        assert_eq!(AsrTask::Transcribe.endpoint(), "/v1/audio/transcriptions");
        assert_eq!(AsrTask::Translate.endpoint(), "/v1/audio/translations");
    }

    #[test]
    fn task_default_is_transcribe() {
        assert_eq!(AsrTask::default(), AsrTask::Transcribe);
    }

    // -----------------------------------------------------------------------
    // decide_task
    // -----------------------------------------------------------------------

    #[test]
    fn decide_task_en_requested_de_detected_translates() {
        assert_eq!(decide_task("en", "de"), TaskDecision::Translate);
    }

    #[test]
    fn decide_task_en_requested_en_detected_transcribes() {
        assert_eq!(
            decide_task("en", "en"),
            TaskDecision::Transcribe {
                language: "en".to_string()
            }
        );
    }

    #[test]
    fn decide_task_fr_requested_fr_detected_transcribes() {
        assert_eq!(
            decide_task("fr", "fr"),
            TaskDecision::Transcribe {
                language: "fr".to_string()
            }
        );
    }

    #[test]
    fn decide_task_fr_requested_de_detected_errors() {
        match decide_task("fr", "de") {
            TaskDecision::Error(msg) => {
                assert!(msg.contains("fr"), "error should mention requested lang");
                assert!(msg.contains("de"), "error should mention detected lang");
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // find_track_by_language
    // -----------------------------------------------------------------------

    fn make_track(index: u32, lang: Option<&str>) -> AudioTrack {
        AudioTrack {
            index,
            codec: "aac".to_string(),
            language: lang.map(|s| s.to_string()),
            title: None,
        }
    }

    #[test]
    fn find_track_matches_eng_for_en() {
        let tracks = vec![
            make_track(1, Some("eng")),
            make_track(2, Some("deu")),
            make_track(3, Some("jpn")),
        ];
        let found = find_track_by_language(&tracks, "en");
        assert_eq!(found.map(|t| t.index), Some(1));
    }

    #[test]
    fn find_track_matches_deu_for_de() {
        let tracks = vec![
            make_track(1, Some("eng")),
            make_track(2, Some("deu")),
            make_track(3, Some("jpn")),
        ];
        let found = find_track_by_language(&tracks, "de");
        assert_eq!(found.map(|t| t.index), Some(2));
    }

    #[test]
    fn find_track_matches_jpn_for_ja() {
        let tracks = vec![
            make_track(1, Some("eng")),
            make_track(2, Some("deu")),
            make_track(3, Some("jpn")),
        ];
        let found = find_track_by_language(&tracks, "ja");
        assert_eq!(found.map(|t| t.index), Some(3));
    }

    #[test]
    fn find_track_no_match_returns_none() {
        let tracks = vec![
            make_track(1, Some("eng")),
            make_track(2, Some("deu")),
        ];
        assert!(find_track_by_language(&tracks, "fr").is_none());
    }

    #[test]
    fn find_track_no_language_tag_returns_none() {
        let tracks = vec![make_track(1, None), make_track(2, None)];
        assert!(find_track_by_language(&tracks, "en").is_none());
    }

    #[test]
    fn find_track_case_insensitive() {
        let tracks = vec![make_track(1, Some("ENG"))];
        assert_eq!(find_track_by_language(&tracks, "en").map(|t| t.index), Some(1));
    }
}
