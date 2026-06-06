use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Multipart, Query, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

use futures_util::TryStreamExt;

use crate::ffmpeg::{self, AudioTrack};
use crate::metrics::{Metrics, stt_upstream_labels};
use crate::node_state::{NodeStates, SharedNodeStates, ordered_upstreams};
use crate::proxy::{error_response, is_transport_failure};
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
    /// Whether this task asks whisper-server to translate to English.
    ///
    /// whisper.cpp exposes translate as a `translate=true` form field on the
    /// single transcription endpoint — there is no `/v1/audio/translations`
    /// route. Translation always targets English.
    fn is_translate(self) -> bool {
        matches!(self, AsrTask::Translate)
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

/// Build the whisper-server transcription URL on a llama-swap upstream:
/// `{base}/upstream/{model}/v1/audio/transcriptions`. The `/upstream/<model>`
/// prefix selects the model by URL (version-proof) and forwards the rest of the
/// path to the child server verbatim.
fn transcription_url(stt_base: &str, model: &str) -> String {
    format!("{stt_base}/upstream/{model}/v1/audio/transcriptions")
}

/// Normalize a language value returned by whisper.cpp (`verbose_json`) into an
/// ISO 639-1 two-letter code.
///
/// whisper.cpp emits the English language NAME (commonly lowercase, e.g.
/// "english"), whereas Speaches emitted the code directly. The rest of the
/// `/asr` pipeline (`decide_task`, `find_track_by_language`, `language_name`)
/// requires the 2-letter code, so this bridges the gap: accept an existing
/// code, then an explicit map of whisper's known outputs, then a
/// case-insensitive `isolang` name lookup. Returns `None` for truly unknown
/// values (the caller treats that as a failed detection rather than guessing).
///
/// NOTE: the exact casing/spelling whisper-server emits is confirmed against a
/// live backend in Unit 6 (RTF/parity gate). This lookup is deliberately
/// tolerant so a casing surprise degrades to a clean failure, never a silent
/// mis-route.
fn normalize_language_code(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_lowercase();

    // Already an ISO 639-1 code?
    if lower.len() == 2 {
        if let Some(lang) = isolang::Language::from_639_1(&lower) {
            return lang.to_639_1().map(str::to_string);
        }
    }

    // Explicit map for whisper.cpp's language names where they may differ from
    // isolang's canonical English name.
    if let Some(code) = whisper_name_to_639_1(&lower) {
        return Some(code.to_string());
    }

    // Fall back to isolang's English-name lookup (canonical names are
    // title-cased, e.g. "English").
    if let Some(code) =
        isolang::Language::from_name(&title_case_words(&lower)).and_then(|lang| lang.to_639_1())
    {
        return Some(code.to_string());
    }
    isolang::Language::from_name(trimmed)
        .and_then(|lang| lang.to_639_1())
        .map(str::to_string)
}

/// Title-case each whitespace-separated word ("modern greek" -> "Modern Greek").
fn title_case_words(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Map whisper.cpp language names (lowercase) to ISO 639-1 codes for common
/// languages, covering cases where isolang's canonical name differs from
/// whisper's output. Returns `None` for unmapped names (the caller falls back
/// to an isolang name lookup).
fn whisper_name_to_639_1(name: &str) -> Option<&'static str> {
    let code = match name {
        "english" => "en",
        "chinese" | "mandarin" => "zh",
        "german" => "de",
        "spanish" | "castilian" => "es",
        "russian" => "ru",
        "korean" => "ko",
        "french" => "fr",
        "japanese" => "ja",
        "portuguese" => "pt",
        "turkish" => "tr",
        "polish" => "pl",
        "catalan" => "ca",
        "dutch" | "flemish" => "nl",
        "arabic" => "ar",
        "swedish" => "sv",
        "italian" => "it",
        "indonesian" => "id",
        "hindi" => "hi",
        "finnish" => "fi",
        "vietnamese" => "vi",
        "hebrew" => "he",
        "ukrainian" => "uk",
        "greek" => "el",
        "malay" => "ms",
        "czech" => "cs",
        "romanian" | "moldavian" | "moldovan" => "ro",
        "danish" => "da",
        "hungarian" => "hu",
        "norwegian" => "no",
        "thai" => "th",
        "urdu" => "ur",
        _ => return None,
    };
    Some(code)
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
fn find_track_by_language<'a>(tracks: &'a [AudioTrack], lang: &str) -> Option<&'a AudioTrack> {
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
    /// Ordered llama-swap STT upstreams; v1 uses the first entry.
    pub stt_upstreams: Vec<String>,
    /// llama-swap model id used in the `/upstream/{model}` path
    /// (transcription + language detection).
    pub stt_model: String,
    /// Model id for the translate-to-English task (whisper turbo can't
    /// translate, so this is a separate non-turbo model).
    pub stt_translate_model: String,
    pub client: std::sync::Arc<reqwest::Client>,
    pub metrics: std::sync::Arc<Metrics>,
    /// Shared node-state snapshot driving state-aware upstream ordering. An
    /// empty snapshot (poller disabled) degrades to the static `stt_upstreams`
    /// order.
    pub node_states: SharedNodeStates,
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

    if state.stt_upstreams.is_empty() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no STT upstream configured",
        );
    }

    // Read the node-state snapshot once for this request; both detection and
    // transcription reorder against it (lock-free, off the hot path).
    // `load_full` clones the inner Arc so it can be held across awaits safely.
    let snapshot = state.node_states.load_full();

    match select_and_transcribe(
        &audio.temp_path,
        &audio.file_name,
        requested_lang,
        params.output,
        response_format,
        &state.stt_model,
        &state.stt_translate_model,
        &state.stt_upstreams,
        &snapshot,
        &state.client,
        &state.metrics,
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
#[allow(clippy::too_many_arguments)]
async fn select_and_transcribe(
    file_path: &Path,
    file_name: &str,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    translate_model: &str,
    stt_bases: &[String],
    states: &NodeStates,
    client: &reqwest::Client,
    metrics: &Metrics,
) -> Result<Response, Response> {
    if ffmpeg::is_video_file(file_name) {
        select_and_transcribe_video(
            file_path,
            requested_lang,
            output,
            response_format,
            model,
            translate_model,
            stt_bases,
            states,
            client,
            metrics,
        )
        .await
    } else {
        select_and_transcribe_audio(
            file_path,
            requested_lang,
            output,
            response_format,
            model,
            translate_model,
            stt_bases,
            states,
            client,
            metrics,
        )
        .await
    }
}

/// Detect the language of an audio file, failing over across `stt_bases` in
/// order: the first upstream that yields a detection wins.
///
/// Detection runs *before* the transcribe/translate call, so if it stayed
/// pinned to the primary, an rh-anine outage would fail `/asr` (Bazarr) at the
/// detection step and never reach the failover-capable transcription. Unlike
/// the transcription failover, this falls through on ANY detection error (not
/// only transport failures): detection is best-effort and has no partial-
/// response risk, so a failure on the primary should still try the backup
/// (plan 2026-06-02-003).
async fn detect_language_with_failover(
    stt_bases: &[String],
    model: &str,
    states: &NodeStates,
    client: &reqwest::Client,
    metrics: &Metrics,
    audio_path: &Path,
) -> Result<String, String> {
    // Reorder the upstream list by current node state (R1–R4); detection uses
    // the STT model. The result is a permutation of `stt_bases`, so every
    // upstream is still tried — state only chooses the starting order (R5).
    let (order, decision) = ordered_upstreams(stt_bases, states, model);
    metrics.record_routing_decision(decision);

    let mut last_err = "no STT upstream configured".to_string();
    for base in &order {
        match detect_language_from_file(base, model, client, audio_path).await {
            Ok(lang) => return Ok(lang),
            Err(e) => {
                tracing::warn!(error = %e, upstream = %base, "language detection failed on upstream, trying next");
                last_err = e;
            }
        }
    }
    Err(last_err)
}

/// Handle a video file: probe tracks, try the language-matched track first,
/// fall back to track 0, then give up.
#[allow(clippy::too_many_arguments)]
async fn select_and_transcribe_video(
    video_path: &Path,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    translate_model: &str,
    stt_bases: &[String],
    states: &NodeStates,
    client: &reqwest::Client,
    metrics: &Metrics,
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

        let detected = detect_language_with_failover(
            stt_bases,
            model,
            states,
            client,
            metrics,
            &audio.temp_path,
        )
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
                return send_with_failover(
                    &audio.temp_path,
                    &language,
                    AsrTask::Transcribe,
                    output,
                    response_format,
                    model,
                    stt_bases,
                    states,
                    client,
                    metrics,
                )
                .await;
            }
            TaskDecision::Translate => {
                // Whisper translate always targets English; pass the detected
                // language as the source, and use the dedicated translate model
                // (turbo can't translate).
                return send_with_failover(
                    &audio.temp_path,
                    &detected,
                    AsrTask::Translate,
                    output,
                    response_format,
                    translate_model,
                    stt_bases,
                    states,
                    client,
                    metrics,
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
#[allow(clippy::too_many_arguments)]
async fn select_and_transcribe_audio(
    audio_path: &Path,
    requested_lang: &str,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    translate_model: &str,
    stt_bases: &[String],
    states: &NodeStates,
    client: &reqwest::Client,
    metrics: &Metrics,
) -> Result<Response, Response> {
    let detected =
        detect_language_with_failover(stt_bases, model, states, client, metrics, audio_path)
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
            send_with_failover(
                audio_path,
                &language,
                AsrTask::Transcribe,
                output,
                response_format,
                model,
                stt_bases,
                states,
                client,
                metrics,
            )
            .await
        }
        TaskDecision::Translate => {
            send_with_failover(
                audio_path,
                &detected,
                AsrTask::Translate,
                output,
                response_format,
                translate_model,
                stt_bases,
                states,
                client,
                metrics,
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
// Speaches request builder + ordered failover
// ---------------------------------------------------------------------------

/// Build a fresh transcription/translation multipart form, reopening the audio
/// file each time. A `reqwest` body stream is single-use, so every failover
/// attempt needs its own `Form` built from a newly-opened file handle.
async fn build_transcription_form(
    audio_path: &Path,
    language: &str,
    task: AsrTask,
    response_format: &str,
    model: &str,
) -> Result<reqwest::multipart::Form, Response> {
    let file = tokio::fs::File::open(audio_path).await.map_err(|e| {
        tracing::warn!(error = %e, "failed to open audio file for stt backend");
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

    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", response_format.to_string())
        .text("temperature", "0.0")
        .text("language", language.to_string());
    if task.is_translate() {
        // whisper.cpp translates to English via this form field; there is no
        // separate /v1/audio/translations route.
        form = form.text("translate", "true");
    }
    Ok(form)
}

/// Send a transcription/translation request to the STT backend, failing over
/// through `stt_bases` in order, and return the formatted HTTP response.
///
/// Failover triggers ONLY on a transport-level failure (see
/// [`is_transport_failure`]): if an upstream returns any HTTP response it is
/// considered alive and that response is surfaced as-is — so a broken-but-
/// responding node is never masked and a llama-swap cold-load 503 is never
/// mistaken for an outage (plan 2026-06-02-003, R3). Each attempt rebuilds the
/// multipart form from a freshly-opened file. A per-upstream `served` /
/// `fell_through` metric records the outcome.
#[allow(clippy::too_many_arguments)]
async fn send_with_failover(
    audio_path: &Path,
    language: &str,
    task: AsrTask,
    output: AsrOutput,
    response_format: &str,
    model: &str,
    stt_bases: &[String],
    states: &NodeStates,
    client: &reqwest::Client,
    metrics: &Metrics,
) -> Result<Response, Response> {
    // Reorder by current node state for the model actually requested (transcribe
    // vs translate may differ). A permutation of `stt_bases` — the loop still
    // iterates every upstream, so the transport-failover floor is unchanged (R5).
    let (order, decision) = ordered_upstreams(stt_bases, states, model);
    metrics.record_routing_decision(decision);
    let last_idx = order.len().saturating_sub(1);

    for (idx, stt_base) in order.iter().enumerate() {
        let is_last = idx == last_idx;
        let form =
            build_transcription_form(audio_path, language, task, response_format, model).await?;
        let url = transcription_url(stt_base, model);

        let record_fell_through = |stt_base: &str| {
            metrics
                .stt_upstream_attempts
                .get_or_create(&stt_upstream_labels(stt_base, "fell_through"))
                .inc();
        };

        match client.post(&url).multipart(form).send().await {
            Ok(upstream) => {
                let status = StatusCode::from_u16(upstream.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                match upstream.bytes().await {
                    Ok(body) => {
                        // Record `served` only once the full response body has
                        // arrived — a send that succeeds but whose body read
                        // fails was not actually served.
                        metrics
                            .stt_upstream_attempts
                            .get_or_create(&stt_upstream_labels(stt_base, "served"))
                            .inc();

                        let content_type = content_type_for(output);
                        let mut response = (status, body).into_response();
                        response.headers_mut().insert(
                            axum::http::header::CONTENT_TYPE,
                            HeaderValue::from_static(content_type),
                        );
                        if output == AsrOutput::Srt {
                            response.headers_mut().insert(
                                axum::http::header::CONTENT_DISPOSITION,
                                HeaderValue::from_static(
                                    "attachment; filename=\"transcription.srt\"",
                                ),
                            );
                        }
                        return Ok(response);
                    }
                    // A mid-body transport failure (upstream accepted the
                    // connection then reset/timed out) is still a dead-upstream
                    // signal. Nothing has been sent to the client yet — the body
                    // is buffered here, not streamed — so fail over like a
                    // connect error rather than returning a 502.
                    Err(e) if is_transport_failure(&e) && !is_last => {
                        tracing::warn!(error = %e, upstream = %stt_base, "stt response read failed mid-body, failing over to next");
                        record_fell_through(stt_base);
                        continue;
                    }
                    Err(e) => {
                        if is_transport_failure(&e) {
                            record_fell_through(stt_base);
                        }
                        tracing::warn!(error = %e, upstream = %stt_base, "failed to read stt backend response body");
                        return Err(error_response(
                            StatusCode::BAD_GATEWAY,
                            "failed to read upstream response",
                        ));
                    }
                }
            }
            Err(e) if is_transport_failure(&e) && !is_last => {
                tracing::warn!(
                    error = %e, upstream = %stt_base,
                    "stt upstream unreachable, failing over to next"
                );
                record_fell_through(stt_base);
                continue;
            }
            Err(e) => {
                // Record the final upstream's transport failure too, so the
                // metric reflects every failed attempt (not just fell-overs).
                if is_transport_failure(&e) {
                    record_fell_through(stt_base);
                }
                tracing::warn!(error = %e, upstream = %stt_base, "stt backend request failed");
                return Err(error_response(
                    StatusCode::BAD_GATEWAY,
                    "upstream unavailable",
                ));
            }
        }
    }

    // Reachable only if stt_bases is empty (handlers guard against this first).
    Err(error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "no STT upstream configured",
    ))
}

// ---------------------------------------------------------------------------
// POST /detect-language
// ---------------------------------------------------------------------------

pub async fn handle_detect_language(
    State(state): State<AsrState>,
    Query(params): Query<AsrParams>,
    multipart: Multipart,
) -> Response {
    let mut audio = match extract_audio_file(multipart).await {
        Ok(a) => a,
        Err(resp) => return resp,
    };

    // When encode=false the upload is raw 16-bit signed-integer PCM at 16 kHz
    // mono (Bazarr's whisperai provider ffmpeg-encodes to headerless s16le and
    // sends encode=false to BOTH /asr and /detect-language). Wrap it in a WAV
    // header so ffmpeg can decode it — without this, detection ffprobe fails
    // with "Invalid data found", Bazarr sees an empty language code, and
    // subtitle generation aborts as "isn't valid for this file" for any media
    // lacking an audio-language tag. Mirrors handle_asr.
    if !params.encode {
        audio = match wrap_raw_pcm_as_wav(audio).await {
            Ok(a) => a,
            Err(resp) => return resp,
        };
    }

    // If the upload is a video container, extract its first audio track.
    let audio = if ffmpeg::is_video_file(&audio.file_name) {
        match extract_video_audio_track(&audio.temp_path, None).await {
            Ok(a) => a,
            Err(resp) => return resp,
        }
    } else {
        audio
    };

    if state.stt_upstreams.is_empty() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no STT upstream configured",
        );
    }

    let snapshot = state.node_states.load_full();
    let code = match detect_language_with_failover(
        &state.stt_upstreams,
        &state.stt_model,
        &snapshot,
        &state.client,
        &state.metrics,
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

/// Max concurrent language-detection requests in flight against a single
/// whisper-server (which serialises inference behind one mutex). Bounding this
/// avoids queueing all chunks at once and tripping per-request timeouts.
const DETECT_CONCURRENCY: usize = 4;

/// Per-request timeout for one detection chunk — generous enough to absorb a
/// llama-swap cold model load (~5-11s) plus a 30s-chunk transcription.
const DETECT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Minimum number of chunk detections that must succeed before trusting the
/// majority vote; below this we fail loudly rather than guess a language.
const DETECT_MIN_QUORUM: usize = 2;

/// Detect the language of an audio file using multi-chunk majority voting.
///
/// This is the reusable entry point called by both the `/asr` pipeline and
/// the `/detect-language` endpoint.
async fn detect_language_from_file(
    stt_base: &str,
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
        return detect_language_single(stt_base, model, client, audio_path).await;
    }

    // Extract segments and detect concurrently, but bound concurrency: a single
    // whisper-server serialises inference behind one mutex, so firing all N
    // chunks at once just queues them and risks per-request timeouts.
    let sem = Arc::new(Semaphore::new(DETECT_CONCURRENCY));
    let mut handles = Vec::with_capacity(offsets.len());

    for offset in &offsets {
        let stt_base = stt_base.to_string();
        let model = model.to_string();
        let client = client.clone();
        let offset = *offset;
        let audio_path = audio_path.to_path_buf();
        let sem = sem.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|e| format!("detection semaphore closed: {e}"))?;

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

            detect_language_single(&stt_base, &model, &client, &segment_path).await
        }));
    }

    let mut votes: HashMap<String, usize> = HashMap::new();
    let mut successes = 0usize;

    for handle in handles {
        match handle.await {
            Ok(Ok(lang)) => {
                successes += 1;
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

    // Require a quorum of successful chunks before trusting the vote, so a
    // busy/slow whisper-server (most chunks dropped) can't silently default to
    // a wrong language.
    let quorum = DETECT_MIN_QUORUM.min(offsets.len());
    if successes < quorum {
        return Err(format!(
            "language detection below quorum: {successes}/{} chunks succeeded (need {quorum})",
            offsets.len()
        ));
    }

    votes
        .into_iter()
        .max_by_key(|(_lang, count)| *count)
        .map(|(lang, _)| lang)
        .ok_or_else(|| "all chunk detections failed".to_string())
}

/// Detect the language of a single audio file by sending it to Speaches.
async fn detect_language_single(
    stt_base: &str,
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
        .text("temperature", "0.0")
        // whisper-server defaults --language to "en"; force auto-detect.
        .text("language", "auto");

    let url = transcription_url(stt_base, model);

    let upstream = client
        .post(&url)
        .timeout(DETECT_REQUEST_TIMEOUT)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("stt request failed: {e}"))?;

    if !upstream.status().is_success() {
        return Err(format!("stt backend returned {}", upstream.status()));
    }

    let body: serde_json::Value = upstream
        .json()
        .await
        .map_err(|e| format!("failed to parse verbose_json: {e}"))?;

    let raw = body
        .get("language")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "no language field in response".to_string())?;

    // whisper.cpp verbose_json returns the language NAME (commonly lowercase,
    // e.g. "english"), not the ISO 639-1 code the rest of the pipeline needs.
    normalize_language_code(raw).ok_or_else(|| format!("unrecognized detected language '{raw}'"))
}

// ---------------------------------------------------------------------------
// Multipart audio extraction  (streams to temp file, no memory limit)
// ---------------------------------------------------------------------------

struct AudioFile {
    temp_path: tempfile::TempPath, // auto-deletes on drop
    file_name: String,
}

async fn extract_audio_file(mut multipart: Multipart) -> Result<AudioFile, Response> {
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::warn!(error = %e, "multipart parse error");
        error_response(StatusCode::BAD_REQUEST, "invalid multipart body")
    })? {
        if field.name() == Some("audio_file") {
            let file_name = field.file_name().unwrap_or("audio.wav").to_string();
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
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to create temp file",
            )
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
    fn task_is_translate() {
        assert!(!AsrTask::Transcribe.is_translate());
        assert!(AsrTask::Translate.is_translate());
    }

    #[test]
    fn task_default_is_transcribe() {
        assert_eq!(AsrTask::default(), AsrTask::Transcribe);
    }

    #[test]
    fn asr_params_encode_defaults_true_and_parses_false() {
        // Bazarr always sends encode=false (it pre-encodes to raw s16le PCM)
        // to BOTH /asr and /detect-language; default must stay true for the
        // OpenAI-style passthrough callers.
        let dflt: AsrParams = serde_urlencoded::from_str("").unwrap();
        assert!(dflt.encode, "encode must default to true");
        let off: AsrParams = serde_urlencoded::from_str("encode=false").unwrap();
        assert!(!off.encode, "encode=false must parse to false");
    }

    /// Regression for the "isn't valid for this file" failure: Bazarr posts
    /// headerless raw s16le PCM with encode=false. `wrap_raw_pcm_as_wav` must
    /// turn it into a WAV that ffmpeg can decode, otherwise /detect-language
    /// ffprobe fails ("Invalid data found"), Bazarr gets an empty language
    /// code, and subtitle generation aborts for any media without an
    /// audio-language tag.
    #[tokio::test]
    async fn wrap_raw_pcm_as_wav_yields_decodable_wav() {
        crate::ffmpeg::init();

        // 2 seconds of 16 kHz mono s16le silence — headerless raw PCM.
        let pcm = vec![0u8; (TARGET_RATE_HZ * 2 * 2) as usize];
        let named = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(named.path(), &pcm).unwrap();
        let input = AudioFile {
            temp_path: named.into_temp_path(),
            file_name: "audio".to_string(),
        };

        let wrapped = wrap_raw_pcm_as_wav(input)
            .await
            .map_err(|_| "wrap_raw_pcm_as_wav returned an error response")
            .expect("raw PCM should wrap into a WAV");

        let dur = crate::ffmpeg::get_duration(&wrapped.temp_path)
            .expect("wrapped WAV must be decodable by ffmpeg");
        assert!(
            (dur - 2.0).abs() < 0.2,
            "wrapped WAV duration should be ~2s, got {dur}"
        );
    }

    /// 16 kHz, named to match the WAV header written by `wrap_raw_pcm_as_wav`.
    const TARGET_RATE_HZ: u32 = 16_000;

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
        let tracks = vec![make_track(1, Some("eng")), make_track(2, Some("deu"))];
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
        assert_eq!(
            find_track_by_language(&tracks, "en").map(|t| t.index),
            Some(1)
        );
    }

    // -----------------------------------------------------------------------
    // transcription_url
    // -----------------------------------------------------------------------

    #[test]
    fn transcription_url_builds_upstream_path() {
        assert_eq!(
            transcription_url("http://llama-swap:8080", "whisper"),
            "http://llama-swap:8080/upstream/whisper/v1/audio/transcriptions"
        );
    }

    // -----------------------------------------------------------------------
    // normalize_language_code  (P0: whisper.cpp returns a NAME, not an ISO code)
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_language_name_lowercase_via_map() {
        assert_eq!(normalize_language_code("english").as_deref(), Some("en"));
        assert_eq!(normalize_language_code("german").as_deref(), Some("de"));
        assert_eq!(normalize_language_code("japanese").as_deref(), Some("ja"));
        assert_eq!(normalize_language_code("dutch").as_deref(), Some("nl"));
    }

    #[test]
    fn normalize_language_titlecased_input() {
        assert_eq!(normalize_language_code("English").as_deref(), Some("en"));
        assert_eq!(normalize_language_code("  German  ").as_deref(), Some("de"));
    }

    #[test]
    fn normalize_language_existing_code_passthrough() {
        assert_eq!(normalize_language_code("en").as_deref(), Some("en"));
        assert_eq!(normalize_language_code("DE").as_deref(), Some("de"));
    }

    #[test]
    fn normalize_language_unknown_returns_none() {
        assert_eq!(normalize_language_code("klingon"), None);
        assert_eq!(normalize_language_code(""), None);
        assert_eq!(normalize_language_code("   "), None);
    }
}

// ---------------------------------------------------------------------------
// Ordered STT failover tests (Unit 3) — spin minimal local mock STT servers
// (axum is already a dependency) to exercise send_with_failover end-to-end.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod failover_tests {
    use super::*;
    use axum::Router;
    use axum::routing::post;
    use prometheus_client::registry::Registry;
    use std::io::Write;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    /// Spawn a mock STT server that records (received body length, hit count)
    /// and replies with `status` + `body`. Returns its base URL.
    async fn spawn_mock(
        status: u16,
        body: &'static str,
    ) -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let received = Arc::new(AtomicUsize::new(0));
        let hits = Arc::new(AtomicUsize::new(0));
        let r = received.clone();
        let h = hits.clone();

        let app = Router::new().route(
            "/upstream/whisper/v1/audio/transcriptions",
            post(move |b: axum::body::Bytes| {
                let r = r.clone();
                let h = h.clone();
                async move {
                    h.fetch_add(1, Ordering::SeqCst);
                    r.store(b.len(), Ordering::SeqCst);
                    axum::response::Response::builder()
                        .status(status)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body))
                        .unwrap()
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), received, hits)
    }

    /// A base URL whose port is bound then released — connecting refuses, which
    /// is the transport failure that triggers failover.
    async fn closed_addr() -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a: SocketAddr = l.local_addr().unwrap();
        drop(l);
        format!("http://{a}")
    }

    fn audio_temp() -> tempfile::TempPath {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 2048]).unwrap();
        f.into_temp_path()
    }

    fn test_metrics() -> Metrics {
        let mut reg = Registry::default();
        Metrics::new(&mut reg)
    }

    fn served(m: &Metrics, base: &str) -> u64 {
        m.stt_upstream_attempts
            .get_or_create(&stt_upstream_labels(base, "served"))
            .get()
    }
    fn fell_through(m: &Metrics, base: &str) -> u64 {
        m.stt_upstream_attempts
            .get_or_create(&stt_upstream_labels(base, "fell_through"))
            .get()
    }

    async fn run(
        bases: &[String],
        client: &reqwest::Client,
        metrics: &Metrics,
    ) -> Result<Response, Response> {
        // Empty snapshot → static ordering (today's behaviour); these tests
        // exercise the unchanged transport-failover floor.
        run_with_states(bases, &NodeStates::new(), client, metrics).await
    }

    async fn run_with_states(
        bases: &[String],
        states: &NodeStates,
        client: &reqwest::Client,
        metrics: &Metrics,
    ) -> Result<Response, Response> {
        let audio = audio_temp();
        send_with_failover(
            &audio,
            "en",
            AsrTask::Transcribe,
            AsrOutput::Json,
            "verbose_json",
            "whisper",
            bases,
            states,
            client,
            metrics,
        )
        .await
    }

    /// Build a NodeStates with the given (base, reachable, ready-models) tuples.
    fn states_of(entries: &[(&str, bool, &[&str])]) -> NodeStates {
        entries
            .iter()
            .map(|(base, reachable, ready)| {
                (
                    (*base).to_string(),
                    crate::node_state::NodeSnapshot {
                        reachable: *reachable,
                        ready_models: ready.iter().map(|s| s.to_string()).collect(),
                    },
                )
            })
            .collect()
    }

    // axum's Response<Body> is not Debug, so Result::expect can't be used.
    fn ok_status(r: Result<Response, Response>) -> StatusCode {
        match r {
            Ok(resp) => resp.status(),
            Err(_) => panic!("expected Ok response, got Err"),
        }
    }
    fn err_status(r: Result<Response, Response>) -> StatusCode {
        match r {
            Ok(_) => panic!("expected Err response, got Ok"),
            Err(resp) => resp.status(),
        }
    }

    #[tokio::test]
    async fn happy_single_upstream_no_failover() {
        let (base, received, hits) = spawn_mock(200, r#"{"text":"hello"}"#).await;
        let metrics = test_metrics();
        let client = reqwest::Client::new();

        let status = ok_status(run(std::slice::from_ref(&base), &client, &metrics).await);

        assert_eq!(status, StatusCode::OK);
        assert!(received.load(Ordering::SeqCst) >= 2048, "got the full body");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(served(&metrics, &base), 1);
        assert_eq!(fell_through(&metrics, &base), 0);
    }

    #[tokio::test]
    async fn fails_over_to_second_upstream_with_full_body() {
        let down = closed_addr().await;
        let (up, received, hits) = spawn_mock(200, r#"{"text":"hi"}"#).await;
        let metrics = test_metrics();
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();

        let status = ok_status(run(&[down.clone(), up.clone()], &client, &metrics).await);

        assert_eq!(status, StatusCode::OK);
        // The key correctness property: the body is re-materialised for the
        // retry, so the second upstream receives the full payload (not empty).
        assert!(
            received.load(Ordering::SeqCst) >= 2048,
            "second upstream must receive the full re-materialised body"
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(fell_through(&metrics, &down), 1);
        assert_eq!(served(&metrics, &up), 1);
    }

    #[tokio::test]
    async fn loading_503_is_not_failed_over() {
        // A 503 is a real HTTP response → the node is alive → return it as-is,
        // do NOT try the backup (don't mistake a cold-load for an outage).
        let (busy, _r, busy_hits) = spawn_mock(503, r#"{"error":"loading"}"#).await;
        let (backup, _br, backup_hits) = spawn_mock(200, r#"{"text":"x"}"#).await;
        let metrics = test_metrics();
        let client = reqwest::Client::new();

        let status = ok_status(run(&[busy.clone(), backup.clone()], &client, &metrics).await);

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy_hits.load(Ordering::SeqCst), 1);
        assert_eq!(
            backup_hits.load(Ordering::SeqCst),
            0,
            "must not fail over on a 503 response"
        );
        assert_eq!(served(&metrics, &busy), 1);
        assert_eq!(fell_through(&metrics, &busy), 0);
    }

    #[tokio::test]
    async fn detection_fails_over_to_second_upstream() {
        use std::io::Write;
        // 1 s of silence as a valid 16 kHz mono 16-bit WAV so ffmpeg can probe
        // its duration and take the single-request detection path.
        crate::ffmpeg::init();
        let pcm = vec![0u8; 32000];
        let header = crate::wyoming::wav_header(pcm.len() as u32, 16000, 1, 2);
        let mut f = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&pcm).unwrap();
        let path = f.into_temp_path();

        let down = closed_addr().await;
        let (up, _recv, _hits) = spawn_mock(200, r#"{"language":"english"}"#).await;
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();

        let bases = vec![down, up];
        let states = NodeStates::new();
        let metrics = test_metrics();
        let lang =
            detect_language_with_failover(&bases, "whisper", &states, &client, &metrics, &path)
                .await
                .expect("detection fails over to the healthy upstream");
        assert_eq!(lang, "en");
    }

    #[tokio::test]
    async fn all_upstreams_down_returns_bad_gateway() {
        let down1 = closed_addr().await;
        let down2 = closed_addr().await;
        let metrics = test_metrics();
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();

        let status = err_status(run(&[down1.clone(), down2.clone()], &client, &metrics).await);

        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(fell_through(&metrics, &down1), 1);
    }

    // -- state-aware reorder (Unit 3) --

    #[tokio::test]
    async fn r2_reorders_to_warm_failover_first() {
        // Primary reachable but model not ready; failover has it warm → the
        // request must hit the warm failover first and skip the primary.
        let (primary, _pr, primary_hits) = spawn_mock(200, r#"{"text":"p"}"#).await;
        let (failover, recv, failover_hits) = spawn_mock(200, r#"{"text":"f"}"#).await;
        let metrics = test_metrics();
        let client = reqwest::Client::new();

        let states = states_of(&[
            (primary.as_str(), true, &[]),
            (failover.as_str(), true, &["whisper-large-v3-turbo"]),
        ]);
        let bases = vec![primary.clone(), failover.clone()];

        let status = ok_status(run_with_states(&bases, &states, &client, &metrics).await);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            failover_hits.load(Ordering::SeqCst),
            1,
            "warm failover must be hit first"
        );
        assert_eq!(
            primary_hits.load(Ordering::SeqCst),
            0,
            "cold primary must not be contacted"
        );
        assert!(
            recv.load(Ordering::SeqCst) >= 2048,
            "full body re-materialised for the reordered upstream"
        );
        assert_eq!(served(&metrics, &failover), 1);
    }

    #[tokio::test]
    async fn r1_primary_ready_keeps_primary_first() {
        // Primary ready → unchanged order: the primary is hit first.
        let (primary, recv, primary_hits) = spawn_mock(200, r#"{"text":"p"}"#).await;
        let (failover, _fr, failover_hits) = spawn_mock(200, r#"{"text":"f"}"#).await;
        let metrics = test_metrics();
        let client = reqwest::Client::new();

        let states = states_of(&[
            (primary.as_str(), true, &["whisper-large-v3-turbo"]),
            (failover.as_str(), true, &["whisper-large-v3-turbo"]),
        ]);
        let bases = vec![primary.clone(), failover.clone()];

        let status = ok_status(run_with_states(&bases, &states, &client, &metrics).await);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            primary_hits.load(Ordering::SeqCst),
            1,
            "ready primary must be hit first"
        );
        assert_eq!(failover_hits.load(Ordering::SeqCst), 0);
        assert!(recv.load(Ordering::SeqCst) >= 2048);
        assert_eq!(served(&metrics, &primary), 1);
    }
}
