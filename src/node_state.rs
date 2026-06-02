//! Node-state layer for state-aware STT routing (plan 2026-06-02-004).
//!
//! A background task polls each STT upstream's llama-swap `/running` endpoint
//! (a live "ps" of loaded models with a per-model `state`) on a fixed interval
//! and publishes an immutable per-node snapshot. The request path reads that
//! snapshot (lock-free, via [`arc_swap`]) and a pure decision function
//! ([`ordered_upstreams`], Unit 2) reorders the upstream list per request.
//!
//! The snapshot is a *hint* layered over the existing ordered transport-failover
//! loop: it only chooses the starting order, so a stale or missing snapshot
//! degrades to exactly today's static-order behaviour and never blocks STT
//! (R5/R6). Polling is fully off the request hot path.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;

/// Per-request timeout for one `/running` poll. `/running` is cheap (it never
/// triggers a model load), so a short timeout keeps a hung node from stalling a
/// poll cycle — a timeout simply marks the node unreachable for that cycle.
const POLL_REQUEST_TIMEOUT: Duration = Duration::from_secs(3);

/// Snapshot of one STT upstream's loaded-model state, as last observed via
/// llama-swap's `/running` endpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeSnapshot {
    /// Whether the last poll reached the node (the GET returned *any* HTTP
    /// response). A transport-level failure (connect refused / timeout) sets
    /// this to `false`.
    pub reachable: bool,
    /// Canonical model ids the node reported in state `ready` at the last poll.
    pub ready_models: HashSet<String>,
}

/// The latest per-node state, keyed by upstream base URL. An entry is absent
/// until the node has been polled at least once.
pub type NodeStates = HashMap<String, NodeSnapshot>;

/// Shared, lock-free-read handle to the current [`NodeStates`] snapshot. The
/// poller swaps in a fresh `Arc<NodeStates>` each cycle; readers `load()` the
/// current one without taking a lock or awaiting.
pub type SharedNodeStates = Arc<ArcSwap<NodeStates>>;

/// Build a fresh shared handle initialised with an empty snapshot. With an
/// empty snapshot, routing degrades to the static configured order.
pub fn new_shared() -> SharedNodeStates {
    Arc::new(ArcSwap::from_pointee(NodeStates::new()))
}

/// Parse a `/running` response body, collecting the canonical model ids whose
/// `state` is exactly `"ready"`.
///
/// Tolerant by design: unknown top-level/entry fields are ignored, a missing or
/// malformed `running` array yields no ready models, and any `state` other than
/// `"ready"` (e.g. `loading`, `stopping`, or an unknown future value) is treated
/// as not-ready. This keeps an llama-swap schema surprise from ever inventing a
/// false "ready" (which would be the only unsafe direction — it could reorder
/// load onto a node that can't actually serve).
fn parse_ready_models(body: &serde_json::Value) -> HashSet<String> {
    let mut ready = HashSet::new();
    let Some(entries) = body.get("running").and_then(|v| v.as_array()) else {
        return ready;
    };
    for entry in entries {
        if entry.get("state").and_then(|v| v.as_str()) == Some("ready") {
            if let Some(model) = entry.get("model").and_then(|v| v.as_str()) {
                ready.insert(model.to_string());
            }
        }
    }
    ready
}

/// Poll a single node's `/running` endpoint and build its snapshot.
///
/// Fail-safe: a transport error marks the node unreachable with no ready models.
/// A response with a non-success status (or an unparseable body) is treated as
/// *reachable but with no known-ready models* — the node answered, so we must
/// not deprioritise it as "down"; the absence of warmth info simply degrades
/// routing to the safe primary-first order for that node.
async fn poll_node(base: &str, client: &reqwest::Client) -> NodeSnapshot {
    let url = format!("{base}/running");
    match client.get(&url).timeout(POLL_REQUEST_TIMEOUT).send().await {
        Ok(resp) => {
            let ready_models = if resp.status().is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(body) => parse_ready_models(&body),
                    Err(e) => {
                        tracing::debug!(error = %e, upstream = %base, "failed to parse /running body");
                        HashSet::new()
                    }
                }
            } else {
                tracing::debug!(status = %resp.status(), upstream = %base, "/running returned non-success status");
                HashSet::new()
            };
            NodeSnapshot {
                reachable: true,
                ready_models,
            }
        }
        Err(e) => {
            tracing::debug!(error = %e, upstream = %base, "/running poll failed; marking node unreachable");
            NodeSnapshot {
                reachable: false,
                ready_models: HashSet::new(),
            }
        }
    }
}

/// Background poll loop: every `interval`, poll every base's `/running`
/// concurrently and publish the combined snapshot. Never returns.
///
/// Spawned only when the interval is non-zero (see [`spawn`]). Each cycle does a
/// single in-flight GET per node; failures are isolated per node and never
/// surface to a request.
pub async fn poll_loop(
    bases: Vec<String>,
    client: Arc<reqwest::Client>,
    shared: SharedNodeStates,
    interval: Duration,
) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        let snaps = futures_util::future::join_all(
            bases
                .iter()
                .map(|base| async { (base.clone(), poll_node(base, &client).await) }),
        )
        .await;
        let states: NodeStates = snaps.into_iter().collect();
        tracing::debug!(?states, "published node-state snapshot");
        shared.store(Arc::new(states));
    }
}

/// Spawn the background poller and return the shared snapshot handle.
///
/// When `interval_secs == 0` the poller is *not* spawned: the returned handle
/// stays an empty snapshot forever, so [`ordered_upstreams`] degrades to the
/// static configured order (the instant escape hatch / rollback knob — R6).
pub fn spawn(
    bases: Vec<String>,
    client: Arc<reqwest::Client>,
    interval_secs: u64,
) -> SharedNodeStates {
    let shared = new_shared();
    if interval_secs == 0 {
        tracing::info!(
            "STATE_POLL_INTERVAL_SECS=0: node-state poller disabled (static STT ordering)"
        );
        return shared;
    }
    tracing::info!(
        interval_secs,
        nodes = ?bases,
        "starting node-state poller for state-aware STT routing"
    );
    tokio::spawn(poll_loop(
        bases,
        client,
        shared.clone(),
        Duration::from_secs(interval_secs),
    ));
    shared
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ready_collects_only_ready_models() {
        let body = serde_json::json!({
            "running": [
                {"model": "whisper-large-v3-turbo", "state": "ready", "ttl": 600},
                {"model": "kokoro", "state": "ready"},
            ]
        });
        let ready = parse_ready_models(&body);
        assert!(ready.contains("whisper-large-v3-turbo"));
        assert!(ready.contains("kokoro"));
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn parse_ready_empty_running_array() {
        let body = serde_json::json!({ "running": [] });
        assert!(parse_ready_models(&body).is_empty());
    }

    #[test]
    fn parse_ready_excludes_loading_and_unknown_states() {
        let body = serde_json::json!({
            "running": [
                {"model": "whisper-large-v3-turbo", "state": "loading"},
                {"model": "other", "state": "stopping"},
                {"model": "future", "state": "some-new-state"},
                {"model": "warm", "state": "ready"},
            ]
        });
        let ready = parse_ready_models(&body);
        assert_eq!(ready.len(), 1);
        assert!(ready.contains("warm"));
    }

    #[test]
    fn parse_ready_tolerates_missing_or_malformed_fields() {
        // No `running` key at all.
        assert!(parse_ready_models(&serde_json::json!({})).is_empty());
        // `running` not an array.
        assert!(parse_ready_models(&serde_json::json!({"running": "nope"})).is_empty());
        // Entry missing `model` / `state`.
        let body = serde_json::json!({
            "running": [
                {"state": "ready"},          // no model id
                {"model": "x"},               // no state
                {"model": "y", "state": "ready", "extra": 1}, // unknown field ok
            ]
        });
        let ready = parse_ready_models(&body);
        assert_eq!(ready.len(), 1);
        assert!(ready.contains("y"));
    }

    // -- poll_node against local mock servers (axum is already a dependency) --

    use axum::Router;
    use axum::routing::get;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    async fn spawn_running_mock(status: u16, body: &'static str) -> String {
        let app = Router::new().route(
            "/running",
            get(move || async move {
                axum::response::Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap()
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn closed_addr() -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a: SocketAddr = l.local_addr().unwrap();
        drop(l);
        format!("http://{a}")
    }

    #[tokio::test]
    async fn poll_node_happy_reports_ready_models() {
        let base = spawn_running_mock(
            200,
            r#"{"running":[{"model":"whisper-large-v3-turbo","state":"ready"}]}"#,
        )
        .await;
        let client = reqwest::Client::new();
        let snap = poll_node(&base, &client).await;
        assert!(snap.reachable);
        assert!(snap.ready_models.contains("whisper-large-v3-turbo"));
    }

    #[tokio::test]
    async fn poll_node_empty_is_reachable_with_no_models() {
        let base = spawn_running_mock(200, r#"{"running":[]}"#).await;
        let client = reqwest::Client::new();
        let snap = poll_node(&base, &client).await;
        assert!(snap.reachable);
        assert!(snap.ready_models.is_empty());
    }

    #[tokio::test]
    async fn poll_node_transport_error_marks_unreachable() {
        let base = closed_addr().await;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let snap = poll_node(&base, &client).await;
        assert!(!snap.reachable);
        assert!(snap.ready_models.is_empty());
    }

    #[tokio::test]
    async fn poll_node_non_success_is_reachable_without_warmth() {
        let base = spawn_running_mock(500, "boom").await;
        let client = reqwest::Client::new();
        let snap = poll_node(&base, &client).await;
        // The node answered, so it is reachable; we just learn no warmth from it.
        assert!(snap.reachable);
        assert!(snap.ready_models.is_empty());
    }
}
