use std::net::SocketAddr;
use tokio::net::TcpListener;

/// A base URL whose port is bound then released — connecting refuses, which
/// is the transport failure that triggers failover.
pub async fn closed_addr() -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a: SocketAddr = l.local_addr().unwrap();
    drop(l);
    format!("http://{a}")
}
