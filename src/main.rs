use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::extract::DefaultBodyLimit;
use axum::routing::{any, get, post};
use axum::Router;
use prometheus_client::registry::Registry;
use serde_json::json;
use tokio::net::TcpListener;
use tracing::info;

use speech_router::asr::{self, AsrState};
use speech_router::config::Config;
use speech_router::metrics::{self, Metrics};
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
        speaches_url = %config.speaches_url,
        public_addr = %config.public_addr,
        wyoming_port = config.wyoming_port,
        internal_addr = %config.internal_addr,
        default_model = %config.default_model,
        "starting speech-router"
    );

    let wyoming_port = config.wyoming_port;
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .build()
            .expect("failed to build HTTP client"),
    );

    let mut registry = Registry::default();
    let metrics = Arc::new(Metrics::new(&mut registry));

    let state = AppState {
        config: Arc::new(config),
        client,
        metrics,
        registry: Arc::new(registry),
    };

    // Wyoming protocol TCP server (STT + TTS via Speaches).
    let wyoming_listener = TcpListener::bind(("0.0.0.0", wyoming_port))
        .await
        .expect("failed to bind Wyoming port");
    tokio::spawn(speech_router::wyoming::serve(
        wyoming_listener,
        state.config.clone(),
        state.client.clone(),
    ));

    let asr_state = AsrState {
        speaches_url: state.config.speaches_url.clone(),
        default_model: state.config.default_model.clone(),
        client: state.client.clone(),
    };

    let asr_router: Router<()> = Router::new()
        .route("/asr", post(asr::handle_asr))
        .route("/detect-language", post(asr::handle_detect_language))
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

    let public_server = axum::serve(public_listener, public_router)
        .with_graceful_shutdown(shutdown_signal());
    let internal_server = axum::serve(internal_listener, internal_router)
        .with_graceful_shutdown(shutdown_signal());

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

    proxy::forward(proxy::ProxyRequest {
        client: &state.client,
        backend_url: &state.config.speaches_url,
        path: uri.path(),
        query: uri.query(),
        method,
        headers: &headers,
        body,
    })
    .await
}

// ---------------------------------------------------------------------------
// Internal handlers
// ---------------------------------------------------------------------------

async fn health_route(State(state): State<AppState>) -> Response {
    let check_url = format!("{}/v1/models", state.config.speaches_url);

    match state.client.get(&check_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            (StatusCode::OK, axum::Json(json!({"status": "ok"}))).into_response()
        }
        Ok(resp) => {
            let status = resp.status();
            (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({
                    "status": "unhealthy",
                    "reason": format!("speaches returned {status}")
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({
                "status": "unhealthy",
                "reason": format!("speaches unreachable: {e}")
            })),
        )
            .into_response(),
    }
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
