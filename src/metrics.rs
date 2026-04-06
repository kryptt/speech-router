use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

const DURATION_BUCKETS: &[f64] = &[
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Prometheus metrics for the speech router.
///
/// All metric fields use interior atomics -- the struct is safe to share
/// behind an `Arc` without additional synchronisation.
#[derive(Clone)]
pub struct Metrics {
    pub requests_total: Family<Vec<(String, String)>, Counter>,
    pub request_duration: Family<Vec<(String, String)>, Histogram>,
    pub wyoming_connections: Gauge,
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

        Metrics {
            requests_total,
            request_duration,
            wyoming_connections,
        }
    }
}

/// Build the label set for `requests_total`.
pub fn request_labels(
    protocol: &str,
    operation: &str,
    status: &str,
) -> Vec<(String, String)> {
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
    fn encode_registry_after_increment() {
        let mut registry = Registry::default();
        let metrics = Metrics::new(&mut registry);

        let labels = request_labels("openai", "transcribe", "ok");
        metrics.requests_total.get_or_create(&labels).inc();

        let output = encode_registry(&registry).expect("encode should succeed");
        assert!(output.contains("speech_router_requests_total"));
    }
}
