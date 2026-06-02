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

// ---------------------------------------------------------------------------
// Pure ordering decision (Unit 2)
// ---------------------------------------------------------------------------

/// The decision label emitted alongside a reordered upstream list, mirroring
/// the rows of the routing decision matrix. Used for the
/// `stt_routing_decisions_total{decision}` metric (Unit 4) so the policy is
/// observable in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Primary has the model `ready` (or we have no state for it) — order
    /// unchanged. This is the steady-state, low-load-on-primary path (R1).
    PrimaryReady,
    /// Primary is reachable but the model is not `ready`, and a failover already
    /// has it `ready` — the warm failover is hoisted first (R2: latency win, no
    /// extra VRAM cost).
    FailoverWarm,
    /// Primary is reachable, model not `ready`, and no failover is warm — keep
    /// primary-first and let it (cold-)load (R4: don't cold-load the contended
    /// failover on a mere primary `loading`).
    PrimaryCold,
    /// Primary was unreachable at the last poll — reachable failovers are
    /// hoisted ahead of it to skip the per-request connect wait (R3).
    PrimaryDown,
    /// No node was reachable at the last poll — order unchanged; the failover
    /// loop surfaces the error (R5 floor).
    AllUnreachable,
}

impl RoutingDecision {
    /// Stable metric label for this decision.
    pub fn as_str(self) -> &'static str {
        match self {
            RoutingDecision::PrimaryReady => "primary_ready",
            RoutingDecision::FailoverWarm => "failover_warm",
            RoutingDecision::PrimaryCold => "primary_cold",
            RoutingDecision::PrimaryDown => "primary_down",
            RoutingDecision::AllUnreachable => "all_unreachable",
        }
    }
}

/// Best-effort match between a requested model id/alias and a canonical id
/// reported by `/running`.
///
/// llama-swap reports the *canonical* id (e.g. `whisper-large-v3-turbo`) while
/// speech-router requests an *alias* (e.g. `whisper`). A match is positive only
/// when we are confident: an exact match, or when the canonical id is the alias
/// extended by dash-delimited version segments. The comparison is segment-wise
/// (not raw substring) so `whisper` matches `whisper-large-v3-turbo` but
/// `whisper-translate` does NOT match `whisper-large-v3-turbo`, and `whisper`
/// does NOT match an unrelated `whisperx`. Any ambiguity therefore resolves to
/// "no match", which degrades routing to the safe primary-first order (the
/// optimisation is lost, never correctness — the failover floor still applies).
fn model_matches(canonical: &str, requested: &str) -> bool {
    if canonical == requested {
        return true;
    }
    let mut canon_segments = canonical.split('-');
    requested
        .split('-')
        .all(|seg| canon_segments.next() == Some(seg))
}

/// Whether a node has the requested model `ready` (best-effort, see
/// [`model_matches`]).
fn model_ready(snap: &NodeSnapshot, model: &str) -> bool {
    snap.ready_models
        .iter()
        .any(|canonical| model_matches(canonical, model))
}

/// Move `bases[idx]` to the front, preserving the relative order of the rest.
fn hoist(bases: &[String], idx: usize) -> Vec<String> {
    let mut order = Vec::with_capacity(bases.len());
    order.push(bases[idx].clone());
    for (i, base) in bases.iter().enumerate() {
        if i != idx {
            order.push(base.clone());
        }
    }
    order
}

/// Choose the starting order of STT upstreams for a request needing `model`,
/// given the latest node-state snapshot. Implements the decision matrix from
/// the plan.
///
/// The result is **always a permutation of `bases`** — no entry is ever dropped
/// (R5) — so the existing ordered transport-failover loop still iterates every
/// upstream; state only chooses where to *start*. The returned
/// [`RoutingDecision`] labels which matrix row fired (for the Unit-4 metric).
///
/// Safety properties:
/// - Fewer than two upstreams, or no state for the primary → static order.
/// - Never hoists a failover on a *mere* primary `loading` (R4): a failover is
///   only promoted when it is itself `ready` (R2) or the primary is unreachable
///   (R3). This keeps steady-state load off the contended CUDA failover (R1).
pub fn ordered_upstreams(
    bases: &[String],
    states: &NodeStates,
    model: &str,
) -> (Vec<String>, RoutingDecision) {
    // Nothing to reorder.
    if bases.len() < 2 {
        return (bases.to_vec(), RoutingDecision::PrimaryReady);
    }

    let primary = &bases[0];
    // No state for the primary (poller disabled, or hasn't polled yet) → treat
    // as steady state and keep the static order.
    let Some(primary_snap) = states.get(primary) else {
        return (bases.to_vec(), RoutingDecision::PrimaryReady);
    };

    // R1: primary has the model ready → steady state, no reorder.
    if model_ready(primary_snap, model) {
        return (bases.to_vec(), RoutingDecision::PrimaryReady);
    }

    if primary_snap.reachable {
        // Primary reachable but model not ready. R2: hoist the first failover
        // that already has the model warm; else R4: keep primary-first and let
        // it cold-load (never cold-load the contended failover on a mere
        // primary `loading`).
        let warm = bases.iter().enumerate().skip(1).find_map(|(i, base)| {
            states
                .get(base)
                .filter(|s| s.reachable && model_ready(s, model))
                .map(|_| i)
        });
        return match warm {
            Some(idx) => (hoist(bases, idx), RoutingDecision::FailoverWarm),
            None => (bases.to_vec(), RoutingDecision::PrimaryCold),
        };
    }

    // Primary unreachable. R3: route to reachable failovers first (preserving
    // their relative order), then the remaining bases (the down primary and any
    // unreachable failovers) in their original order. If none are reachable,
    // keep the static order and let the loop surface the error.
    let reachable_failovers: Vec<&String> = bases
        .iter()
        .skip(1)
        .filter(|base| states.get(*base).is_some_and(|s| s.reachable))
        .collect();
    if reachable_failovers.is_empty() {
        return (bases.to_vec(), RoutingDecision::AllUnreachable);
    }
    let mut order: Vec<String> = reachable_failovers.iter().map(|b| (*b).clone()).collect();
    for base in bases {
        if !reachable_failovers.contains(&base) {
            order.push(base.clone());
        }
    }
    (order, RoutingDecision::PrimaryDown)
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

    // -----------------------------------------------------------------------
    // model_matches  (alias↔canonical, safe under ambiguity)
    // -----------------------------------------------------------------------

    #[test]
    fn model_matches_exact_and_alias_prefix() {
        // exact
        assert!(model_matches("whisper", "whisper"));
        // alias is a dash-delimited prefix of the canonical id
        assert!(model_matches("whisper-large-v3-turbo", "whisper"));
        assert!(model_matches("whisper-large-v3", "whisper"));
    }

    #[test]
    fn model_matches_is_segment_wise_not_substring() {
        // `whisper-translate` must NOT match the turbo canonical id.
        assert!(!model_matches(
            "whisper-large-v3-turbo",
            "whisper-translate"
        ));
        // `whisper` must NOT match an unrelated `whisperx` (segment, not prefix).
        assert!(!model_matches("whisperx", "whisper"));
    }

    // -----------------------------------------------------------------------
    // ordered_upstreams  (the decision matrix — the core correctness surface)
    // -----------------------------------------------------------------------

    fn bases() -> Vec<String> {
        vec!["http://primary".to_string(), "http://failover".to_string()]
    }

    /// Build a NodeStates entry helper.
    fn snap(reachable: bool, ready: &[&str]) -> NodeSnapshot {
        NodeSnapshot {
            reachable,
            ready_models: ready.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn states(entries: &[(&str, NodeSnapshot)]) -> NodeStates {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn r1_primary_ready_keeps_order() {
        let b = bases();
        let s = states(&[
            ("http://primary", snap(true, &["whisper-large-v3-turbo"])),
            ("http://failover", snap(true, &["whisper-large-v3-turbo"])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(order, b);
        assert_eq!(decision, RoutingDecision::PrimaryReady);
    }

    #[test]
    fn r2_primary_not_ready_failover_warm_hoists_failover() {
        let b = bases();
        let s = states(&[
            ("http://primary", snap(true, &[])), // reachable, loading/cold
            ("http://failover", snap(true, &["whisper-large-v3-turbo"])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(
            order,
            vec!["http://failover".to_string(), "http://primary".to_string()]
        );
        assert_eq!(decision, RoutingDecision::FailoverWarm);
    }

    #[test]
    fn r4_primary_not_ready_no_warm_failover_keeps_primary_first() {
        let b = bases();
        let s = states(&[
            ("http://primary", snap(true, &[])),
            ("http://failover", snap(true, &[])), // reachable but also cold
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(order, b, "must NOT cold-load the contended failover");
        assert_eq!(decision, RoutingDecision::PrimaryCold);
    }

    #[test]
    fn r3_primary_unreachable_hoists_reachable_failover() {
        let b = bases();
        let s = states(&[
            ("http://primary", snap(false, &[])),
            ("http://failover", snap(true, &[])), // reachable, even if cold
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(
            order,
            vec!["http://failover".to_string(), "http://primary".to_string()]
        );
        assert_eq!(decision, RoutingDecision::PrimaryDown);
    }

    #[test]
    fn all_unreachable_keeps_order() {
        let b = bases();
        let s = states(&[
            ("http://primary", snap(false, &[])),
            ("http://failover", snap(false, &[])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(order, b);
        assert_eq!(decision, RoutingDecision::AllUnreachable);
    }

    #[test]
    fn empty_snapshot_keeps_static_order() {
        let b = bases();
        let empty = NodeStates::new();
        for model in ["whisper", "whisper-translate", "anything"] {
            let (order, decision) = ordered_upstreams(&b, &empty, model);
            assert_eq!(order, b, "disabled poller must not reorder");
            assert_eq!(decision, RoutingDecision::PrimaryReady);
        }
    }

    #[test]
    fn single_upstream_never_reorders() {
        let b = vec!["http://only".to_string()];
        let s = states(&[("http://only", snap(false, &[]))]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(order, b);
        assert_eq!(decision, RoutingDecision::PrimaryReady);
    }

    #[test]
    fn translate_model_unaffected_by_transcribe_warmth() {
        // Failover has the turbo (transcribe) model ready but we request the
        // translate model: model_matches must not falsely treat that as warm,
        // so we keep primary-first (cold-load on primary).
        let b = bases();
        let s = states(&[
            ("http://primary", snap(true, &[])),
            ("http://failover", snap(true, &["whisper-large-v3-turbo"])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper-translate");
        assert_eq!(order, b);
        assert_eq!(decision, RoutingDecision::PrimaryCold);
    }

    #[test]
    fn three_upstreams_warm_node_hoisted_others_keep_relative_order() {
        let b = vec![
            "http://primary".to_string(),
            "http://fail-a".to_string(),
            "http://fail-b".to_string(),
        ];
        // Primary reachable-but-cold; fail-b is the warm one.
        let s = states(&[
            ("http://primary", snap(true, &[])),
            ("http://fail-a", snap(true, &[])),
            ("http://fail-b", snap(true, &["whisper-large-v3-turbo"])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        assert_eq!(
            order,
            vec![
                "http://fail-b".to_string(),
                "http://primary".to_string(),
                "http://fail-a".to_string(),
            ]
        );
        assert_eq!(decision, RoutingDecision::FailoverWarm);
    }

    #[test]
    fn three_upstreams_primary_down_reachable_failovers_first() {
        let b = vec![
            "http://primary".to_string(),
            "http://fail-a".to_string(),
            "http://fail-b".to_string(),
        ];
        // Primary down, fail-a unreachable too, fail-b reachable.
        let s = states(&[
            ("http://primary", snap(false, &[])),
            ("http://fail-a", snap(false, &[])),
            ("http://fail-b", snap(true, &[])),
        ]);
        let (order, decision) = ordered_upstreams(&b, &s, "whisper");
        // Reachable failover first; the rest keep original order.
        assert_eq!(
            order,
            vec![
                "http://fail-b".to_string(),
                "http://primary".to_string(),
                "http://fail-a".to_string(),
            ]
        );
        assert_eq!(decision, RoutingDecision::PrimaryDown);
    }

    /// R5 property: the output is always a permutation of the input, for every
    /// reachability/warmth combination across two upstreams and both models.
    #[test]
    fn output_is_always_a_permutation() {
        let b = bases();
        let bools = [false, true];
        for &p_reach in &bools {
            for p_ready in [vec![], vec!["whisper-large-v3-turbo"]] {
                for &f_reach in &bools {
                    for f_ready in [vec![], vec!["whisper-large-v3-turbo"]] {
                        let s = states(&[
                            ("http://primary", snap(p_reach, &p_ready)),
                            ("http://failover", snap(f_reach, &f_ready)),
                        ]);
                        for model in ["whisper", "whisper-translate"] {
                            let (order, _decision) = ordered_upstreams(&b, &s, model);
                            let mut sorted = order.clone();
                            sorted.sort();
                            let mut expected = b.clone();
                            expected.sort();
                            assert_eq!(
                                sorted, expected,
                                "order must be a permutation of bases (p_reach={p_reach}, f_reach={f_reach}, model={model})"
                            );
                        }
                    }
                }
            }
        }
    }
}
