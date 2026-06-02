use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

const DURATION_BUCKETS: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0];

/// Prometheus metrics for the speech router.
///
/// All metric fields use interior atomics -- the struct is safe to share
/// behind an `Arc` without additional synchronisation.
#[derive(Clone)]
pub struct Metrics {
    pub requests_total: Family<Vec<(String, String)>, Counter>,
    pub request_duration: Family<Vec<(String, String)>, Histogram>,
    pub wyoming_connections: Gauge,
    /// Incremented whenever a Wyoming STT request fails and an empty transcript
    /// is returned as a fallback. A non-zero rate signals a (silent-to-the-user)
    /// STT backend outage and is the trigger for investigating/rolling back.
    pub stt_empty_transcript_fallback: Counter,
    /// Per-upstream STT attempt outcomes, labeled by `upstream` (the base URL)
    /// and `outcome` (`served` | `fell_through`). A non-zero `fell_through` on
    /// the primary upstream is the signal that the primary node is unreachable
    /// and traffic is failing over (plan 2026-06-02-003, R3) — it must not be
    /// allowed to silently mask a broken node.
    pub stt_upstream_attempts: Family<Vec<(String, String)>, Counter>,
    /// State-aware routing decisions, labeled by `decision` (`primary_ready` |
    /// `failover_warm` | `primary_cold` | `primary_down` | `all_unreachable`).
    /// A non-zero `failover_warm`/`primary_down` rate is the signal that
    /// state-aware routing is actively reordering; a `primary_ready`-dominated
    /// steady state confirms the CUDA failover takes no steady-state load
    /// (plan 2026-06-02-004, R7). Fixed cardinality (5 series).
    pub stt_routing_decisions: Family<Vec<(String, String)>, Counter>,
}

impl Metrics {
    /// Register all metrics with `speech_router_` prefix on `registry`.
    pub fn new(registry: &mut Registry) -> Self {
        let requests_total = Family::default();
        registry.register(
            "speech_router_requests_total",
            "Total speech-router requests by protocol, operation, and status",
            requests_total.clone(),
        );

        fn make_duration_histogram() -> Histogram {
            Histogram::new(DURATION_BUCKETS.iter().copied())
        }
        let request_duration =
            Family::new_with_constructor(make_duration_histogram as fn() -> Histogram);
        registry.register(
            "speech_router_request_duration_seconds",
            "Request duration in seconds by protocol and operation",
            request_duration.clone(),
        );

        let wyoming_connections: Gauge = Gauge::default();
        registry.register(
            "speech_router_wyoming_connections",
            "Current number of active Wyoming TCP connections",
            wyoming_connections.clone(),
        );

        let stt_empty_transcript_fallback: Counter = Counter::default();
        registry.register(
            "speech_router_stt_empty_transcript_fallback_total",
            "Wyoming STT requests that failed and returned an empty transcript fallback",
            stt_empty_transcript_fallback.clone(),
        );

        let stt_upstream_attempts = Family::default();
        registry.register(
            "speech_router_stt_upstream_attempts_total",
            "STT upstream attempts by upstream URL and outcome (served | fell_through)",
            stt_upstream_attempts.clone(),
        );

        let stt_routing_decisions = Family::default();
        registry.register(
            "speech_router_stt_routing_decisions_total",
            "State-aware STT routing decisions by chosen-order decision",
            stt_routing_decisions.clone(),
        );

        Metrics {
            requests_total,
            request_duration,
            wyoming_connections,
            stt_empty_transcript_fallback,
            stt_upstream_attempts,
            stt_routing_decisions,
        }
    }

    /// Record one state-aware routing decision (Unit 4 / R7).
    pub fn record_routing_decision(&self, decision: crate::node_state::RoutingDecision) {
        self.stt_routing_decisions
            .get_or_create(&routing_decision_labels(decision.as_str()))
            .inc();
    }
}

/// Build the label set for `stt_upstream_attempts`.
pub fn stt_upstream_labels(upstream: &str, outcome: &str) -> Vec<(String, String)> {
    vec![
        ("upstream".to_owned(), upstream.to_owned()),
        ("outcome".to_owned(), outcome.to_owned()),
    ]
}

/// Build the label set for `stt_routing_decisions`.
pub fn routing_decision_labels(decision: &str) -> Vec<(String, String)> {
    vec![("decision".to_owned(), decision.to_owned())]
}

/// Build the label set for `requests_total`.
pub fn request_labels(protocol: &str, operation: &str, status: &str) -> Vec<(String, String)> {
    vec![
        ("protocol".to_owned(), protocol.to_owned()),
        ("operation".to_owned(), operation.to_owned()),
        ("status".to_owned(), status.to_owned()),
    ]
}

/// Encode the full `registry` into Prometheus text exposition format.
pub fn encode_registry(registry: &Registry) -> Result<String, std::fmt::Error> {
    let mut buf = String::new();
    encode(&mut buf, registry)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_new_does_not_panic() {
        let mut registry = Registry::default();
        let _metrics = Metrics::new(&mut registry);
    }

    #[test]
    fn request_labels_produces_correct_vec() {
        let labels = request_labels("asr", "transcribe", "ok");
        assert_eq!(
            labels,
            vec![
                ("protocol".to_owned(), "asr".to_owned()),
                ("operation".to_owned(), "transcribe".to_owned()),
                ("status".to_owned(), "ok".to_owned()),
            ]
        );
    }

    #[test]
    fn routing_decision_labels_produces_correct_vec() {
        assert_eq!(
            routing_decision_labels("failover_warm"),
            vec![("decision".to_owned(), "failover_warm".to_owned())]
        );
    }

    #[test]
    fn routing_decision_metric_encodes_each_series() {
        let mut registry = Registry::default();
        let metrics = Metrics::new(&mut registry);
        metrics.record_routing_decision(crate::node_state::RoutingDecision::FailoverWarm);
        metrics.record_routing_decision(crate::node_state::RoutingDecision::PrimaryDown);
        metrics.record_routing_decision(crate::node_state::RoutingDecision::PrimaryDown);

        let output = encode_registry(&registry).expect("encode should succeed");
        assert!(output.contains("speech_router_stt_routing_decisions_total"));
        assert!(output.contains("decision=\"failover_warm\""));
        assert!(output.contains("decision=\"primary_down\""));
    }

    #[test]
    fn encode_registry_after_increment() {
        let mut registry = Registry::default();
        let metrics = Metrics::new(&mut registry);

        let labels = request_labels("openai", "transcribe", "ok");
        metrics.requests_total.get_or_create(&labels).inc();

        let output = encode_registry(&registry).expect("encode should succeed");
        assert!(output.contains("speech_router_requests_total"));
    }
}
