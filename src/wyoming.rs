use std::sync::Arc;

use bytes::BytesMut;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

use crate::config::Config;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Header line that precedes every Wyoming event on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventHeader {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data_length: usize,
    pub payload_length: usize,
}

/// A fully parsed Wyoming event: header + optional JSON data + optional binary
/// payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub header: EventHeader,
    /// JSON data bytes (length == header.data_length).
    pub data: Vec<u8>,
    /// Binary payload bytes (length == header.payload_length).
    pub payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Reading / writing events
// ---------------------------------------------------------------------------

/// Read one Wyoming event from a buffered stream.
///
/// Returns `Ok(None)` on clean EOF (connection closed).
async fn read_event<R: AsyncBufReadExt + Unpin>(reader: &mut R) -> std::io::Result<Option<Event>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        return Ok(None);
    }

    let header: EventHeader = serde_json::from_str(line.trim_end_matches('\n'))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut data = vec![0u8; header.data_length];
    if header.data_length > 0 {
        reader.read_exact(&mut data).await?;
    }

    let mut payload = vec![0u8; header.payload_length];
    if header.payload_length > 0 {
        reader.read_exact(&mut payload).await?;
    }

    Ok(Some(Event {
        header,
        data,
        payload,
    }))
}

/// Write one Wyoming event to a stream.
async fn write_event<W: AsyncWriteExt + Unpin>(writer: &mut W, event: &Event) -> std::io::Result<()> {
    let header_json = serde_json::to_string(&event.header)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    writer.write_all(header_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;

    if !event.data.is_empty() {
        writer.write_all(&event.data).await?;
    }
    if !event.payload.is_empty() {
        writer.write_all(&event.payload).await?;
    }

    writer.flush().await?;
    Ok(())
}

/// Build a Wyoming event from parts.
fn make_event(event_type: &str, data: Option<&serde_json::Value>, payload: &[u8]) -> Event {
    let data_bytes = data
        .map(|v| serde_json::to_vec(v).unwrap_or_default())
        .unwrap_or_default();

    Event {
        header: EventHeader {
            event_type: event_type.to_string(),
            data_length: data_bytes.len(),
            payload_length: payload.len(),
        },
        data: data_bytes,
        payload: payload.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// WAV header
// ---------------------------------------------------------------------------

/// Build a standard 44-byte RIFF/WAV header for raw PCM data.
///
/// This is intentionally hand-rolled rather than using a crate like `hound`.
/// The RIFF/WAV header is a fixed 44-byte structure with a well-known layout;
/// pulling in `hound::WavWriter` would require writing PCM data through it
/// just to produce the same bytes, adding more code than it saves.
///
/// This is a pure function with no side effects -- suitable for unit testing.
pub(crate) fn wav_header(pcm_len: u32, rate: u32, channels: u16, width: u16) -> [u8; 44] {
    let bits_per_sample = width * 8;
    let block_align = channels * width;
    let byte_rate = rate * u32::from(block_align);
    let file_size = 36 + pcm_len; // total file size minus 8 for RIFF header

    let mut h = [0u8; 44];
    let mut w = &mut h[..];

    // RIFF chunk
    copy_into(&mut w, b"RIFF");
    copy_into(&mut w, &file_size.to_le_bytes());
    copy_into(&mut w, b"WAVE");

    // fmt sub-chunk
    copy_into(&mut w, b"fmt ");
    copy_into(&mut w, &16u32.to_le_bytes()); // sub-chunk size
    copy_into(&mut w, &1u16.to_le_bytes()); // PCM format
    copy_into(&mut w, &channels.to_le_bytes());
    copy_into(&mut w, &rate.to_le_bytes());
    copy_into(&mut w, &byte_rate.to_le_bytes());
    copy_into(&mut w, &block_align.to_le_bytes());
    copy_into(&mut w, &bits_per_sample.to_le_bytes());

    // data sub-chunk
    copy_into(&mut w, b"data");
    copy_into(&mut w, &pcm_len.to_le_bytes());

    h
}

/// Helper: copy `src` into the front of `dst` and advance the slice.
fn copy_into(dst: &mut &mut [u8], src: &[u8]) {
    dst[..src.len()].copy_from_slice(src);
    let tmp = std::mem::take(dst);
    *dst = &mut tmp[src.len()..];
}

// ---------------------------------------------------------------------------
// Info / describe response
// ---------------------------------------------------------------------------

fn info_event(config: &Config) -> Event {
    let info = serde_json::json!({
        "asr": [{
            "name": "speaches",
            "description": "Speaches STT",
            "installed": true,
            "attribution": {
                "name": "speaches",
                "url": "https://speaches.ai"
            },
            "models": [{
                "name": config.default_model,
                "description": format!("Faster Whisper {}", config.default_model),
                "installed": true,
                "languages": ["en", "es", "fr", "de", "nl"]
            }]
        }],
        "tts": [{
            "name": "speaches-tts",
            "description": "Speaches TTS",
            "installed": true,
            "models": [{
                "name": config.default_tts_model,
                "description": format!("{} TTS", config.default_tts_model),
                "installed": true,
                "languages": ["en"]
            }]
        }]
    });

    make_event("info", Some(&info), &[])
}

// ---------------------------------------------------------------------------
// STT handler
// ---------------------------------------------------------------------------

/// Accumulating state for an in-progress STT transcription.
struct SttSession {
    language: String,
    rate: u32,
    width: u16,
    channels: u16,
    pcm: BytesMut,
}

impl SttSession {
    fn new(language: String) -> Self {
        Self {
            language,
            rate: 16000,
            width: 2,
            channels: 1,
            pcm: BytesMut::new(),
        }
    }
}

/// Handle `audio-stop`: build WAV, POST to Speaches, return transcript event.
async fn finish_stt(
    session: &SttSession,
    config: &Config,
    client: &reqwest::Client,
) -> Result<Event, String> {
    let pcm_len = session.pcm.len() as u32;
    let header = wav_header(pcm_len, session.rate, session.channels, session.width);

    let mut wav = Vec::with_capacity(44 + session.pcm.len());
    wav.extend_from_slice(&header);
    wav.extend_from_slice(&session.pcm);

    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("mime error: {e}"))?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", config.default_model.clone())
        .text("language", session.language.clone())
        .text("response_format", "json");

    let url = format!("{}/v1/audio/transcriptions", config.speaches_url);

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("speaches request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("speaches returned {status}: {body}"));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("invalid JSON from speaches: {e}"))?;

    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let data = serde_json::json!({"text": text});
    Ok(make_event("transcript", Some(&data), &[]))
}

// ---------------------------------------------------------------------------
// TTS handler
// ---------------------------------------------------------------------------

const TTS_SAMPLE_RATE: u32 = 22050;
const TTS_WIDTH: u16 = 2;
const TTS_CHANNELS: u16 = 1;
const TTS_CHUNK_SIZE: usize = 8192;

async fn handle_tts<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    text: &str,
    voice: &str,
    config: &Config,
    client: &reqwest::Client,
) -> Result<(), String> {
    let url = format!("{}/v1/audio/speech", config.speaches_url);

    let body = serde_json::json!({
        "model": config.default_tts_model,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
        "sample_rate": TTS_SAMPLE_RATE,
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("speaches TTS request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(format!("speaches TTS returned {status}: {body_text}"));
    }

    let audio_fmt = serde_json::json!({
        "rate": TTS_SAMPLE_RATE,
        "width": TTS_WIDTH,
        "channels": TTS_CHANNELS,
    });

    // audio-start
    write_event(writer, &make_event("audio-start", Some(&audio_fmt), &[]))
        .await
        .map_err(|e| format!("write error: {e}"))?;

    // Stream response body in chunks
    let pcm_bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("failed to read TTS response: {e}"))?;

    let mut remaining = &pcm_bytes[..];
    while !remaining.is_empty() {
        let chunk_len = remaining.len().min(TTS_CHUNK_SIZE);
        let (chunk, rest) = remaining.split_at(chunk_len);
        remaining = rest;

        write_event(writer, &make_event("audio-chunk", Some(&audio_fmt), chunk))
            .await
            .map_err(|e| format!("write error: {e}"))?;
    }

    // audio-stop
    write_event(writer, &make_event("audio-stop", None, &[]))
        .await
        .map_err(|e| format!("write error: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-connection handler
// ---------------------------------------------------------------------------

async fn handle_connection(stream: TcpStream, config: Arc<Config>, client: Arc<reqwest::Client>) {
    let addr = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".into());
    info!(%addr, "wyoming connection accepted");

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let mut stt_session: Option<SttSession> = None;

    loop {
        let event = match read_event(&mut reader).await {
            Ok(Some(ev)) => ev,
            Ok(None) => {
                tracing::debug!(%addr, "wyoming connection closed");
                break;
            }
            Err(e) => {
                warn!(%addr, error = %e, "wyoming read error");
                break;
            }
        };

        let event_type = event.header.event_type.as_str();
        tracing::debug!(%addr, event_type, "wyoming event received");

        match event_type {
            // -- Describe / Info --
            "describe" => {
                let info = info_event(&config);
                if let Err(e) = write_event(&mut write_half, &info).await {
                    warn!(%addr, error = %e, "failed to send info event");
                    break;
                }
            }

            // -- STT flow --
            "transcribe" => {
                let language = if event.data.is_empty() {
                    "en".to_string()
                } else {
                    let v: serde_json::Value =
                        serde_json::from_slice(&event.data).unwrap_or_default();
                    v.get("language")
                        .and_then(|l| l.as_str())
                        .unwrap_or("en")
                        .to_string()
                };
                stt_session = Some(SttSession::new(language));
            }

            "audio-start" => {
                if let Some(ref mut session) = stt_session {
                    if !event.data.is_empty() {
                        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&event.data) {
                            if let Some(r) = v.get("rate").and_then(|r| r.as_u64()) {
                                session.rate = r as u32;
                            }
                            if let Some(w) = v.get("width").and_then(|w| w.as_u64()) {
                                session.width = w as u16;
                            }
                            if let Some(c) = v.get("channels").and_then(|c| c.as_u64()) {
                                session.channels = c as u16;
                            }
                        }
                    }
                    session.pcm.clear();
                }
            }

            "audio-chunk" => {
                if let Some(ref mut session) = stt_session {
                    session.pcm.extend_from_slice(&event.payload);
                }
            }

            "audio-stop" => {
                if let Some(ref session) = stt_session {
                    match finish_stt(session, &config, &client).await {
                        Ok(transcript) => {
                            if let Err(e) = write_event(&mut write_half, &transcript).await {
                                warn!(%addr, error = %e, "failed to send transcript");
                                break;
                            }
                        }
                        Err(e) => {
                            warn!(%addr, error = %e, "STT transcription failed");
                            // Send empty transcript on error so the client is not stuck.
                            let fallback_data = serde_json::json!({"text": ""});
                            let fallback = make_event("transcript", Some(&fallback_data), &[]);
                            if let Err(e) = write_event(&mut write_half, &fallback).await {
                                warn!(%addr, error = %e, "failed to send fallback transcript");
                                break;
                            }
                        }
                    }
                    stt_session = None;
                }
            }

            // -- TTS flow --
            "synthesize" => {
                let (text, voice) = if event.data.is_empty() {
                    (String::new(), config.default_tts_voice.clone())
                } else {
                    let v: serde_json::Value =
                        serde_json::from_slice(&event.data).unwrap_or_default();
                    let text = v
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    let voice = v
                        .get("voice")
                        .and_then(|v| v.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or(&config.default_tts_voice)
                        .to_string();
                    (text, voice)
                };

                if let Err(e) =
                    handle_tts(&mut write_half, &text, &voice, &config, &client).await
                {
                    warn!(%addr, error = %e, "TTS synthesis failed");
                }
            }

            other => {
                tracing::debug!(%addr, event_type = other, "ignoring unknown wyoming event");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the Wyoming protocol TCP server.
///
/// This function never returns under normal operation.
pub async fn serve(listener: TcpListener, config: Arc<Config>, client: Arc<reqwest::Client>) {
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".into());
    info!(%addr, "wyoming server listening");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let config = Arc::clone(&config);
                let client = Arc::clone(&client);
                tokio::spawn(handle_connection(stream, config, client));
            }
            Err(e) => {
                warn!(error = %e, "wyoming accept error");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -- wav_header tests --

    #[test]
    fn wav_header_produces_44_bytes() {
        let h = wav_header(1000, 16000, 1, 2);
        assert_eq!(h.len(), 44);
    }

    #[test]
    fn wav_header_riff_magic() {
        let h = wav_header(0, 16000, 1, 2);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(&h[36..40], b"data");
    }

    #[test]
    fn wav_header_16khz_mono_16bit() {
        let h = wav_header(32000, 16000, 1, 2);

        // channels = 1 (offset 22)
        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 1);
        // rate = 16000 (offset 24)
        assert_eq!(u32::from_le_bytes([h[24], h[25], h[26], h[27]]), 16000);
        // byte_rate = 16000 * 1 * 2 = 32000 (offset 28)
        assert_eq!(u32::from_le_bytes([h[28], h[29], h[30], h[31]]), 32000);
        // block_align = 1 * 2 = 2 (offset 32)
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 2);
        // bits_per_sample = 16 (offset 34)
        assert_eq!(u16::from_le_bytes([h[34], h[35]]), 16);
        // data size (offset 40)
        assert_eq!(u32::from_le_bytes([h[40], h[41], h[42], h[43]]), 32000);
        // file size = 36 + pcm_len (offset 4)
        assert_eq!(u32::from_le_bytes([h[4], h[5], h[6], h[7]]), 36 + 32000);
    }

    #[test]
    fn wav_header_stereo_48khz() {
        let h = wav_header(0, 48000, 2, 2);

        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 2);
        assert_eq!(u32::from_le_bytes([h[24], h[25], h[26], h[27]]), 48000);
        // byte_rate = 48000 * 2 * 2 = 192000
        assert_eq!(u32::from_le_bytes([h[28], h[29], h[30], h[31]]), 192000);
        // block_align = 2 * 2 = 4
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 4);
    }

    // -- EventHeader serde tests --

    #[test]
    fn event_header_deserialize() {
        let json = r#"{"type":"describe","data_length":0,"payload_length":0}"#;
        let header: EventHeader = serde_json::from_str(json).unwrap();
        assert_eq!(header.event_type, "describe");
        assert_eq!(header.data_length, 0);
        assert_eq!(header.payload_length, 0);
    }

    #[test]
    fn event_header_serialize() {
        let header = EventHeader {
            event_type: "transcript".to_string(),
            data_length: 42,
            payload_length: 100,
        };
        let json = serde_json::to_string(&header).unwrap();
        // Verify it round-trips
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "transcript");
        assert_eq!(parsed["data_length"], 42);
        assert_eq!(parsed["payload_length"], 100);
    }

    #[test]
    fn event_header_with_data() {
        let json = r#"{"type":"audio-chunk","data_length":35,"payload_length":8192}"#;
        let header: EventHeader = serde_json::from_str(json).unwrap();
        assert_eq!(header.event_type, "audio-chunk");
        assert_eq!(header.data_length, 35);
        assert_eq!(header.payload_length, 8192);
    }

    // -- Event round-trip test --

    #[tokio::test]
    async fn event_round_trip() {
        let original = Event {
            header: EventHeader {
                event_type: "audio-chunk".to_string(),
                data_length: 13,
                payload_length: 4,
            },
            data: br#"{"rate":16000}"#[..13].to_vec(),
            payload: vec![0x01, 0x02, 0x03, 0x04],
        };

        // Write to buffer
        let mut buf = Vec::new();
        write_event(&mut buf, &original).await.unwrap();

        // Read back
        let mut reader = BufReader::new(Cursor::new(buf));
        let restored = read_event(&mut reader).await.unwrap().unwrap();

        assert_eq!(restored.header, original.header);
        assert_eq!(restored.data, original.data);
        assert_eq!(restored.payload, original.payload);
    }

    #[tokio::test]
    async fn event_round_trip_no_data_no_payload() {
        let original = make_event("audio-stop", None, &[]);

        let mut buf = Vec::new();
        write_event(&mut buf, &original).await.unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let restored = read_event(&mut reader).await.unwrap();

        assert_eq!(restored, Some(original));
    }

    #[tokio::test]
    async fn event_round_trip_with_json_data() {
        let data = serde_json::json!({"text": "hello world"});
        let original = make_event("transcript", Some(&data), &[]);

        let mut buf = Vec::new();
        write_event(&mut buf, &original).await.unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let restored = read_event(&mut reader).await.unwrap().unwrap();

        assert_eq!(restored, original);

        // Verify the data is valid JSON with the expected text
        let parsed: serde_json::Value = serde_json::from_slice(&restored.data).unwrap();
        assert_eq!(parsed["text"], "hello world");
    }

    #[tokio::test]
    async fn read_event_eof_returns_none() {
        let mut reader = BufReader::new(Cursor::new(Vec::<u8>::new()));
        let result = read_event(&mut reader).await.unwrap();
        assert!(result.is_none());
    }
}
