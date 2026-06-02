use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

/// Everything needed to proxy a request to the Speaches backend.
pub struct ProxyRequest<'a> {
    pub client: &'a reqwest::Client,
    pub backend_url: &'a str,
    pub path: &'a str,
    pub query: Option<&'a str>,
    pub method: Method,
    pub headers: &'a HeaderMap,
    pub body: bytes::Bytes,
}

/// Forward a request to the backend and stream the response back.
///
/// Any send failure is mapped to a 502 — callers that need to distinguish a
/// transport failure (to fail over) should use [`try_forward`] instead.
pub async fn forward(req: ProxyRequest<'_>) -> Response {
    let url = format!("{}{}", req.backend_url, req.path);
    match try_forward(req).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::warn!(error = %e, url = %url, "upstream request failed");
            error_response(StatusCode::BAD_GATEWAY, "upstream unavailable")
        }
    }
}

/// Forward a request, returning `Err(reqwest::Error)` if the *send* itself
/// failed so the caller can inspect it (e.g. [`is_transport_failure`]) and
/// decide whether to retry on another upstream. A successful send — even one
/// that yields a 5xx status — returns `Ok` with the streamed response.
pub async fn try_forward(req: ProxyRequest<'_>) -> Result<Response, reqwest::Error> {
    let mut url = format!("{}{}", req.backend_url, req.path);
    if let Some(q) = req.query {
        url.push('?');
        url.push_str(q);
    }

    let mut builder = req.client.request(req.method, &url);

    for (key, value) in req.headers.iter() {
        match key.as_str() {
            "host" | "connection" | "transfer-encoding" | "keep-alive" | "upgrade" => continue,
            _ => builder = builder.header(key.clone(), value.clone()),
        }
    }

    let upstream_resp = builder.body(req.body).send().await?;

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut response_headers = HeaderMap::new();
    for (key, value) in upstream_resp.headers().iter() {
        match key.as_str() {
            "content-type" | "content-disposition" | "content-length" => {
                response_headers.insert(key.clone(), value.clone());
            }
            _ => {}
        }
    }

    let stream = upstream_resp
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;

    Ok(response)
}

/// Build a JSON error response.
pub fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({ "error": message });
    (status, axum::Json(body)).into_response()
}

/// Whether a failed `reqwest` send means "this upstream is down — try the next".
///
/// Only transport-level failures qualify: a refused/timed-out *connection* means
/// the node is unreachable. If the upstream returned ANY HTTP response (even a
/// 5xx, including a llama-swap cold-load 503), it is alive — we surface that
/// response rather than masking it behind failover. This is the failover trigger
/// predicate (plan 2026-06-02-003, R3): don't hide a broken-but-responding node,
/// and don't mistake a model cold-load for an outage.
pub fn is_transport_failure(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}
