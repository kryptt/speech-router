use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use prometheus_client::registry::Registry;
use serde_json::json;
use tokio::net::TcpListener;
use tracing::info;

use speech_router::asr::{self, AsrState};
use speech_router::config::Config;
use speech_router::metrics::{self, Metrics};
use speech_router::node_state::{self, SharedNodeStates};
use speech_router::proxy;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    client: Arc<reqwest::Client>,
    metrics: Arc<Metrics>,
    registry: Arc<Registry>,
    /// Shared node-state snapshot for state-aware STT routing (empty when the
    /// poller is disabled → static ordering).
    node_states: SharedNodeStates,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "speech_router=info".into()),
        )
        .init();

    speech_router::ffmpeg::init();

    let config = Config::from_env().expect("invalid configuration");

    info!(
        stt_upstreams = ?config.stt_upstreams,
        tts_url = %config.tts_url,
        stt_model = %config.stt_model,
        speaches_url = ?config.speaches_url,
        public_addr = %config.public_addr,
        wyoming_port = config.wyoming_port,
        internal_addr = %config.internal_addr,
        "starting speech-router"
    );

    let wyoming_port = config.wyoming_port;
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            // Bound connection establishment so a dead/unreachable backend fails
            // fast. We deliberately do NOT set a total request timeout here:
            // full-file transcription (Bazarr) can legitimately run for minutes.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client"),
    );

    let mut registry = Registry::default();
    let metrics = Arc::new(Metrics::new(&mut registry));

    // Spawn the background node-state poller (if enabled) and hold the shared
    // snapshot handle. Reads are lock-free and off the request hot path; when
    // STATE_POLL_INTERVAL_SECS=0 the snapshot stays empty and routing is static.
    let node_states = node_state::spawn(
        config.stt_upstreams.clone(),
        client.clone(),
        config.state_poll_interval_secs,
    );

    let state = AppState {
        config: Arc::new(config),
        client,
        metrics,
        registry: Arc::new(registry),
        node_states,
    };

    // Wyoming protocol TCP server (STT + TTS via Speaches).
    let wyoming_listener = TcpListener::bind(("0.0.0.0", wyoming_port))
        .await
        .expect("failed to bind Wyoming port");
    tokio::spawn(speech_router::wyoming::serve(
        wyoming_listener,
        state.config.clone(),
        state.node_states.clone(),
        state.client.clone(),
        state.metrics.clone(),
    ));

    let asr_state = AsrState {
        stt_upstreams: state.config.stt_upstreams.clone(),
        stt_model: state.config.stt_model.clone(),
        stt_translate_model: state.config.stt_translate_model.clone(),
        client: state.client.clone(),
        metrics: state.metrics.clone(),
        node_states: state.node_states.clone(),
    };

    let asr_router: Router<()> = Router::new()
        .route("/asr", post(asr::handle_asr))
        .route("/detect-language", post(asr::handle_detect_language))
        .route(
            "/v1/audio/transcriptions",
            post(asr::handle_openai_transcriptions),
        )
        .route("/status", get(status_route))
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024 * 1024)) // 4 GiB — raw PCM of full movies
        .with_state(asr_state);

    let passthrough_router: Router<()> = Router::new()
        .fallback(any(passthrough_route))
        .with_state(state.clone());

    let public_router = passthrough_router.merge(asr_router);

    let internal_router = Router::new()
        .route("/health", get(health_route))
        .route("/metrics", get(metrics_route))
        .with_state(state.clone());

    let public_listener = TcpListener::bind(state.config.public_addr)
        .await
        .expect("failed to bind public port");
    let internal_listener = TcpListener::bind(state.config.internal_addr)
        .await
        .expect("failed to bind internal port");

    info!(
        "listening on {} (public) and {} (internal)",
        state.config.public_addr, state.config.internal_addr
    );

    let public_server =
        axum::serve(public_listener, public_router).with_graceful_shutdown(shutdown_signal());
    let internal_server =
        axum::serve(internal_listener, internal_router).with_graceful_shutdown(shutdown_signal());

    let (r1, r2) = tokio::join!(public_server.into_future(), internal_server.into_future(),);

    if let Err(e) = r1 {
        tracing::error!(error = %e, "public server error");
    }
    if let Err(e) = r2 {
        tracing::error!(error = %e, "internal server error");
    }
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

async fn shutdown_signal() {
    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to install SIGTERM handler")
        .recv()
        .await;
    info!("received SIGTERM, shutting down");
}

// ---------------------------------------------------------------------------
// Public handlers -- passthrough to Speaches
// ---------------------------------------------------------------------------

async fn passthrough_route(
    State(state): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let labels = metrics::request_labels("openai", "passthrough", "ok");
    state.metrics.requests_total.get_or_create(&labels).inc();

    let path = uri.path();

    // Synthesize /v1/models locally so OpenAI-compatible audio clients can
    // validate the connection without a Speaches backend.
    if path == "/v1/models" {
        return models_response(&state.config);
    }

    // whisper-server (reached via /upstream) only translates via a
    // `translate=true` multipart form field, which this byte-level passthrough
    // cannot inject without buffering and re-encoding the form. Rather than
    // silently return a transcription for a translation request, reject it
    // clearly. Audio translation is available via the /asr endpoint (translate
    // task), which sets the field correctly. (Full OpenAI-passthrough
    // translation support is a deferred enhancement.)
    if path == "/v1/audio/translations" {
        return proxy::error_response(
            StatusCode::NOT_IMPLEMENTED,
            "translation via /v1/audio/translations is not supported by this backend; \
             use the /asr endpoint (translate task) for audio translation",
        );
    }

    // TTS goes to Kokoro; anything else falls back to the legacy Speaches URL
    // during the phase-out window.
    let (backend_url, fwd_path): (String, String) = match path {
        "/v1/audio/speech" => (state.config.tts_url.clone(), path.to_string()),
        _ => match &state.config.speaches_url {
            Some(u) => (u.clone(), path.to_string()),
            None => {
                return proxy::error_response(
                    StatusCode::BAD_GATEWAY,
                    "no upstream configured for this path",
                );
            }
        },
    };

    proxy::forward(proxy::ProxyRequest {
        client: &state.client,
        backend_url: &backend_url,
        path: &fwd_path,
        query: uri.query(),
        method,
        headers: &headers,
        body,
    })
    .await
}

/// Synthesize a minimal OpenAI `/v1/models` listing for the audio models so
/// OpenAI-compatible clients (e.g. Open WebUI) can validate the connection
/// without a Speaches backend.
fn models_response(config: &Config) -> Response {
    let models = json!({
        "object": "list",
        "data": [
            { "id": config.stt_model, "object": "model", "owned_by": "speech-router" },
            { "id": config.default_tts_model, "object": "model", "owned_by": "speech-router" },
        ]
    });
    (StatusCode::OK, axum::Json(models)).into_response()
}

// ---------------------------------------------------------------------------
// Internal handlers
// ---------------------------------------------------------------------------

async fn health_route(State(state): State<AppState>) -> Response {
    // Probe the STT backends' llama-swap model lists (which do NOT trigger a
    // model load) so readiness reflects STT availability without depending on
    // Speaches. TTS (Kokoro) is intentionally not gated here.
    //
    // Healthy if ANY configured upstream responds: speech-router can serve STT
    // via failover, so readiness must not be gated on the primary alone. If it
    // were, a primary STT upstream outage would fail the readiness probe and
    // pull speech-router out of its Service endpoints — a total STT outage that
    // *defeats* the failover. We only report unhealthy when every upstream is
    // unreachable.
    if state.config.stt_upstreams.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "unhealthy", "reason": "no STT upstream configured"})),
        )
            .into_response();
    }

    let mut reasons: Vec<String> = Vec::new();
    for base in &state.config.stt_upstreams {
        let check_url = format!("{base}/v1/models");
        match state
            .client
            .get(&check_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                return (StatusCode::OK, axum::Json(json!({"status": "ok"}))).into_response();
            }
            Ok(resp) => reasons.push(format!("{base} returned {}", resp.status())),
            Err(e) => reasons.push(format!("{base} unreachable: {e}")),
        }
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(json!({
            "status": "unhealthy",
            "reason": format!("all STT upstreams unavailable: {}", reasons.join("; "))
        })),
    )
        .into_response()
}

async fn status_route() -> Response {
    (
        StatusCode::OK,
        axum::Json(json!({ "version": env!("CARGO_PKG_VERSION") })),
    )
        .into_response()
}

async fn metrics_route(State(state): State<AppState>) -> Response {
    match metrics::encode_registry(&state.registry) {
        Ok(buf) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            )],
            buf,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

// ---------------------------------------------------------------------------
// OpenAI transcription failover tests (Unit 3) — the /v1/audio/transcriptions
// path used by Open WebUI and Hermes.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod passthrough_failover_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use prometheus_client::registry::Registry;
    use speech_router::asr::AsrState;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    async fn closed_addr() -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a: SocketAddr = l.local_addr().unwrap();
        drop(l);
        format!("http://{a}")
    }

    async fn spawn_mock(status: u16) -> (String, Arc<AtomicUsize>) {
        let received = Arc::new(AtomicUsize::new(0));
        let r = received.clone();
        let app = axum::Router::new().route(
            "/upstream/whisper/v1/audio/transcriptions",
            post(move |b: Bytes| {
                let r = r.clone();
                async move {
                    r.store(b.len(), Ordering::SeqCst);
                    axum::response::Response::builder()
                        .status(status)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(r#"{"text":"ok"}"#))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), received)
    }

    fn asr_state(bases: Vec<String>) -> AsrState {
        let mut registry = Registry::default();
        let metrics = Arc::new(Metrics::new(&mut registry));
        let client = Arc::new(
            reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(2))
                .build()
                .unwrap(),
        );
        AsrState {
            stt_upstreams: bases,
            stt_model: "whisper".to_string(),
            stt_translate_model: "whisper".to_string(),
            client,
            metrics,
            node_states: node_state::new_shared(),
        }
    }

    fn asr_state_with_reachability(bases: Vec<String>, reachable: &[bool]) -> AsrState {
        let state = asr_state(bases.clone());
        let snap: speech_router::node_state::NodeStates = bases
            .iter()
            .zip(reachable.iter())
            .map(|(base, &r)| {
                (
                    base.clone(),
                    speech_router::node_state::NodeSnapshot {
                        reachable: r,
                        ready_models: Default::default(),
                    },
                )
            })
            .collect();
        state.node_states.store(Arc::new(snap));
        state
    }

    fn test_router(state: AsrState) -> Router {
        Router::new()
            .route(
                "/v1/audio/transcriptions",
                post(speech_router::asr::handle_openai_transcriptions),
            )
            .with_state(state)
    }

    /// Minimal valid WAV wrapped in a multipart form body.
    fn wav_multipart() -> (String, Vec<u8>) {
        let boundary = "testboundary";
        // 44-byte WAV header, 0 data samples — valid container.
        let wav: &[u8] = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\
            \x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00\
            data\x00\x00\x00\x00";
        let header = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        );
        let footer = format!("\r\n--{boundary}--\r\n");
        let mut body = Vec::new();
        body.extend_from_slice(header.as_bytes());
        body.extend_from_slice(wav);
        body.extend_from_slice(footer.as_bytes());
        (boundary.to_string(), body)
    }

    fn transcription_req(boundary: &str, body: Vec<u8>) -> Request<Body> {
        Request::post("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn transcription_fails_over_to_second_upstream() {
        let down = closed_addr().await;
        let (up, received) = spawn_mock(200).await;
        let state = asr_state(vec![down.clone(), up.clone()]);
        let fell = state.metrics.clone();
        let router = test_router(state);

        let (b, body) = wav_multipart();
        let resp = router.oneshot(transcription_req(&b, body)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            received.load(Ordering::SeqCst) > 0,
            "second upstream must receive the request"
        );
        // Verify fell_through was recorded for the down upstream.
        let ft = fell
            .stt_upstream_attempts
            .get_or_create(&metrics::stt_upstream_labels(&down, "fell_through"))
            .get();
        assert_eq!(ft, 1);
    }

    #[tokio::test]
    async fn transcription_all_down_is_bad_gateway() {
        let down1 = closed_addr().await;
        let down2 = closed_addr().await;
        let state = asr_state(vec![down1.clone(), down2.clone()]);
        let fell = state.metrics.clone();
        let router = test_router(state);

        let (b, body) = wav_multipart();
        let resp = router.oneshot(transcription_req(&b, body)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let ft1 = fell
            .stt_upstream_attempts
            .get_or_create(&metrics::stt_upstream_labels(&down1, "fell_through"))
            .get();
        let ft2 = fell
            .stt_upstream_attempts
            .get_or_create(&metrics::stt_upstream_labels(&down2, "fell_through"))
            .get();
        assert_eq!(ft1, 1);
        assert_eq!(ft2, 1);
    }

    #[tokio::test]
    async fn transcription_503_is_not_failed_over() {
        let (busy, busy_recv) = spawn_mock(503).await;
        let (backup, backup_recv) = spawn_mock(200).await;
        let state = asr_state(vec![busy.clone(), backup.clone()]);
        let router = test_router(state);

        let (b, body) = wav_multipart();
        let resp = router.oneshot(transcription_req(&b, body)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(busy_recv.load(Ordering::SeqCst) > 0);
        assert_eq!(
            backup_recv.load(Ordering::SeqCst),
            0,
            "a 503 must not trigger failover"
        );
    }

    #[tokio::test]
    async fn transcription_reorders_when_primary_marked_down() {
        let primary = closed_addr().await;
        let (failover, received) = spawn_mock(200).await;
        let state =
            asr_state_with_reachability(vec![primary.clone(), failover.clone()], &[false, true]);
        let fell = state.metrics.clone();
        let router = test_router(state);

        let (b, body) = wav_multipart();
        let resp = router.oneshot(transcription_req(&b, body)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            received.load(Ordering::SeqCst) > 0,
            "reordered failover received the request"
        );
        let ft = fell
            .stt_upstream_attempts
            .get_or_create(&metrics::stt_upstream_labels(&primary, "fell_through"))
            .get();
        assert_eq!(ft, 0, "down primary must not be contacted");
    }
}
