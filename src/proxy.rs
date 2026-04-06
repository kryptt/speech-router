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
pub async fn forward(req: ProxyRequest<'_>) -> Response {
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

    let upstream_resp = match builder.body(req.body).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, url = %url, "upstream request failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                "upstream unavailable",
            );
        }
    };

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

    response
}

/// Build a JSON error response.
pub fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({ "error": message });
    (status, axum::Json(body)).into_response()
}
