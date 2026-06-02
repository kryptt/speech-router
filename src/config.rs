use std::env;
use std::fmt;
use std::net::SocketAddr;

/// Validated, immutable configuration.
/// All parsing happens in `from_env`; once constructed, every field is trusted.
#[derive(Debug, Clone)]
pub struct Config {
    /// Ordered STT upstreams — each a llama-swap base URL (e.g.
    /// `http://llama-swap.ai:8080`). Requests are built as
    /// `{upstream}/upstream/{stt_model}/v1/audio/transcriptions`. v1 ships a
    /// single entry; the list is ordered for the future CUDA-node failover.
    pub stt_upstreams: Vec<String>,
    /// TTS upstream base URL (Kokoro-FastAPI), e.g. `http://kokoro-fastapi.ai:8080`.
    pub tts_url: String,
    /// The llama-swap model id used in the `/upstream/{model}` STT path
    /// (transcription + language detection).
    pub stt_model: String,
    /// Model id for the translate-to-English task. whisper turbo is
    /// transcription-only, so translation routes to a separate (non-turbo)
    /// model. Defaults to `stt_model` when unset.
    pub stt_translate_model: String,
    /// Legacy Speaches base URL, retained only as a fallback for any upstream
    /// left unset during the phase-out window. `None` once fully migrated.
    pub speaches_url: Option<String>,
    pub public_addr: SocketAddr,
    pub wyoming_port: u16,
    pub internal_addr: SocketAddr,
    pub default_tts_model: String,
    pub default_tts_voice: String,
    /// Interval, in seconds, between background `/running` polls that drive
    /// state-aware STT routing. `0` disables the poller entirely, so routing
    /// falls back to the static `STT_URLS` order (the instant rollback knob).
    /// Defaults to 5.
    pub state_poll_interval_secs: u64,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        // Legacy single-backend URL, kept as a fallback during the Speaches
        // phase-out. Empty string is treated as unset.
        let speaches_url = env::var("SPEACHES_URL")
            .ok()
            .map(|s| normalize_url(&s))
            .filter(|s| !s.is_empty());

        // Ordered STT upstreams from STT_URLS (comma-separated). Falls back to
        // the legacy SPEACHES_URL as a single entry; errors if neither is set.
        let stt_upstreams = parse_url_list("STT_URLS");
        let stt_upstreams = if stt_upstreams.is_empty() {
            match &speaches_url {
                Some(u) => vec![u.clone()],
                None => return Err(ConfigError::MissingRequired("STT_URLS")),
            }
        } else {
            stt_upstreams
        };

        // TTS upstream (Kokoro-FastAPI). Falls back to legacy SPEACHES_URL.
        let tts_url = env::var("TTS_URL")
            .ok()
            .map(|s| normalize_url(&s))
            .filter(|s| !s.is_empty())
            .or_else(|| speaches_url.clone())
            .ok_or(ConfigError::MissingRequired("TTS_URL"))?;

        let public_port = parse_env_u16("PUBLIC_PORT", 8000)?;
        let wyoming_port = parse_env_u16("WYOMING_PORT", 10300)?;
        let internal_port = parse_env_u16("INTERNAL_PORT", 9090)?;

        // STT_MODEL is the `/upstream/{id}` model id; DEFAULT_MODEL is the
        // historical name and is honoured as a fallback.
        let stt_model = env::var("STT_MODEL")
            .ok()
            .or_else(|| env::var("DEFAULT_MODEL").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "whisper".to_string());
        let stt_translate_model = env::var("STT_TRANSLATE_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| stt_model.clone());
        let default_tts_model =
            env::var("DEFAULT_TTS_MODEL").unwrap_or_else(|_| "kokoro".to_string());
        let default_tts_voice =
            env::var("DEFAULT_TTS_VOICE").unwrap_or_else(|_| "af_heart".to_string());

        // State-aware-routing poll interval. `0` disables the poller (static
        // ordering); any other value is the seconds between `/running` polls.
        let state_poll_interval_secs = parse_env_u64("STATE_POLL_INTERVAL_SECS", 5)?;

        Ok(Config {
            stt_upstreams,
            tts_url,
            stt_model,
            stt_translate_model,
            speaches_url,
            public_addr: SocketAddr::from(([0, 0, 0, 0], public_port)),
            wyoming_port,
            internal_addr: SocketAddr::from(([0, 0, 0, 0], internal_port)),
            default_tts_model,
            default_tts_voice,
            state_poll_interval_secs,
        })
    }
}

/// Trim trailing slashes from a URL so path concatenation is predictable.
fn normalize_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

/// Parse a comma-separated env var into an ordered list of normalized URLs,
/// dropping empty entries. Returns an empty vec if the var is unset/empty.
fn parse_url_list(key: &str) -> Vec<String> {
    env::var(key)
        .unwrap_or_default()
        .split(',')
        .map(normalize_url)
        .filter(|s| !s.is_empty())
        .collect()
}

#[derive(Debug)]
pub enum ConfigError {
    MissingRequired(&'static str),
    InvalidValue { key: &'static str, value: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequired(key) => {
                write!(f, "{key} is required but not set")
            }
            Self::InvalidValue { key, value } => {
                write!(f, "{key} must be a valid port number, got '{value}'")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn parse_env_u16(key: &'static str, default: u16) -> Result<u16, ConfigError> {
    match env::var(key) {
        Ok(val) => val
            .parse()
            .map_err(|_| ConfigError::InvalidValue { key, value: val }),
        Err(_) => Ok(default),
    }
}

fn parse_env_u64(key: &'static str, default: u64) -> Result<u64, ConfigError> {
    match env::var(key) {
        Ok(val) => val
            .parse()
            .map_err(|_| ConfigError::InvalidValue { key, value: val }),
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a closure with temporary env vars, restoring originals afterward.
    ///
    /// # Safety
    /// Tests are run single-threaded (`cargo test -- --test-threads=1` in CI)
    /// so concurrent env mutation is not a concern.
    fn with_env<F: FnOnce()>(vars: &[(&str, &str)], f: F) {
        let originals: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();

        for (k, v) in vars {
            // SAFETY: tests run single-threaded; no concurrent env readers.
            unsafe { env::set_var(k, v) };
        }

        f();

        for (k, original) in &originals {
            // SAFETY: tests run single-threaded; no concurrent env readers.
            match original {
                Some(v) => unsafe { env::set_var(k, v) },
                None => unsafe { env::remove_var(k) },
            }
        }
    }

    /// Env vars that influence `from_env`, cleared at the start of each test so
    /// one test's settings never leak into another (tests run single-threaded).
    const ALL_KEYS: &[&str] = &[
        "STT_URLS",
        "TTS_URL",
        "STT_MODEL",
        "STT_TRANSLATE_MODEL",
        "SPEACHES_URL",
        "PUBLIC_PORT",
        "WYOMING_PORT",
        "INTERNAL_PORT",
        "DEFAULT_MODEL",
        "DEFAULT_TTS_MODEL",
        "DEFAULT_TTS_VOICE",
        "STATE_POLL_INTERVAL_SECS",
    ];

    fn clear_all() {
        for key in ALL_KEYS {
            // SAFETY: tests run single-threaded; no concurrent env readers.
            unsafe { env::remove_var(key) };
        }
    }

    #[test]
    fn from_env_with_defaults() {
        with_env(
            &[
                ("STT_URLS", "http://llama-swap:8080"),
                ("TTS_URL", "http://kokoro:8080"),
            ],
            || {
                clear_all();
                // re-set the two required vars cleared above
                unsafe {
                    env::set_var("STT_URLS", "http://llama-swap:8080");
                    env::set_var("TTS_URL", "http://kokoro:8080");
                }

                let config = Config::from_env().expect("should parse with defaults");
                assert_eq!(
                    config.stt_upstreams,
                    vec!["http://llama-swap:8080".to_string()]
                );
                assert_eq!(config.tts_url, "http://kokoro:8080");
                assert_eq!(config.stt_model, "whisper");
                // translate model defaults to stt_model when unset
                assert_eq!(config.stt_translate_model, "whisper");
                assert_eq!(config.speaches_url, None);
                assert_eq!(config.public_addr.port(), 8000);
                assert_eq!(config.wyoming_port, 10300);
                assert_eq!(config.internal_addr.port(), 9090);
                assert_eq!(config.default_tts_model, "kokoro");
                assert_eq!(config.default_tts_voice, "af_heart");
                // state-aware-routing poll interval defaults to 5s
                assert_eq!(config.state_poll_interval_secs, 5);
            },
        );
    }

    #[test]
    fn state_poll_interval_default_override_and_disable() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", "http://llama-swap:8080");
                env::set_var("TTS_URL", "http://kokoro:8080");
            }
            // default when unset
            assert_eq!(
                Config::from_env().expect("parse").state_poll_interval_secs,
                5
            );
            // explicit override
            unsafe { env::set_var("STATE_POLL_INTERVAL_SECS", "10") };
            assert_eq!(
                Config::from_env().expect("parse").state_poll_interval_secs,
                10
            );
            // 0 disables the poller (static ordering)
            unsafe { env::set_var("STATE_POLL_INTERVAL_SECS", "0") };
            assert_eq!(
                Config::from_env().expect("parse").state_poll_interval_secs,
                0
            );
        });
    }

    #[test]
    fn state_poll_interval_invalid_is_error() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", "http://llama-swap:8080");
                env::set_var("TTS_URL", "http://kokoro:8080");
                env::set_var("STATE_POLL_INTERVAL_SECS", "not_a_number");
            }
            let err = Config::from_env().unwrap_err();
            assert!(matches!(
                err,
                ConfigError::InvalidValue {
                    key: "STATE_POLL_INTERVAL_SECS",
                    ..
                }
            ));
        });
    }

    #[test]
    fn from_env_with_all_vars() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", "http://a:8080/,http://b:8080");
                env::set_var("TTS_URL", "http://kokoro:8080/");
                env::set_var("STT_MODEL", "whisper-large-v3-turbo");
                env::set_var("PUBLIC_PORT", "7000");
                env::set_var("WYOMING_PORT", "7001");
                env::set_var("INTERNAL_PORT", "7002");
                env::set_var("DEFAULT_TTS_MODEL", "piper");
                env::set_var("DEFAULT_TTS_VOICE", "en_US");
            }
            let config = Config::from_env().expect("should parse all vars");
            // ordered list, trailing slashes trimmed
            assert_eq!(
                config.stt_upstreams,
                vec!["http://a:8080".to_string(), "http://b:8080".to_string()]
            );
            assert_eq!(config.tts_url, "http://kokoro:8080");
            assert_eq!(config.stt_model, "whisper-large-v3-turbo");
            assert_eq!(config.public_addr.port(), 7000);
            assert_eq!(config.wyoming_port, 7001);
            assert_eq!(config.internal_addr.port(), 7002);
            assert_eq!(config.default_tts_model, "piper");
            assert_eq!(config.default_tts_voice, "en_US");
        });
    }

    #[test]
    fn stt_translate_model_override_and_fallback() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", "http://llama-swap:8080");
                env::set_var("TTS_URL", "http://kokoro:8080");
                env::set_var("STT_MODEL", "whisper");
                env::set_var("STT_TRANSLATE_MODEL", "whisper-translate");
            }
            let c = Config::from_env().expect("parse");
            assert_eq!(c.stt_model, "whisper");
            assert_eq!(c.stt_translate_model, "whisper-translate");
            // unset -> falls back to stt_model
            unsafe { env::remove_var("STT_TRANSLATE_MODEL") };
            let c2 = Config::from_env().expect("parse");
            assert_eq!(c2.stt_translate_model, "whisper");
        });
    }

    #[test]
    fn legacy_speaches_url_fallback() {
        with_env(&[("SPEACHES_URL", "x")], || {
            clear_all();
            unsafe { env::set_var("SPEACHES_URL", "http://localhost:8080/") };
            let config = Config::from_env().expect("legacy SPEACHES_URL should satisfy both");
            assert_eq!(
                config.stt_upstreams,
                vec!["http://localhost:8080".to_string()]
            );
            assert_eq!(config.tts_url, "http://localhost:8080");
            assert_eq!(
                config.speaches_url,
                Some("http://localhost:8080".to_string())
            );
        });
    }

    #[test]
    fn missing_all_upstreams_is_error() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            let err = Config::from_env().unwrap_err();
            assert!(matches!(err, ConfigError::MissingRequired("STT_URLS")));
        });
    }

    #[test]
    fn missing_tts_when_only_stt_set_is_error() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe { env::set_var("STT_URLS", "http://llama-swap:8080") };
            let err = Config::from_env().unwrap_err();
            assert!(matches!(err, ConfigError::MissingRequired("TTS_URL")));
        });
    }

    #[test]
    fn empty_stt_urls_entries_are_ignored() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", " , http://a:8080 ,, ");
                env::set_var("TTS_URL", "http://kokoro:8080");
            }
            let config = Config::from_env().expect("should parse");
            assert_eq!(config.stt_upstreams, vec!["http://a:8080".to_string()]);
        });
    }

    #[test]
    fn invalid_port_is_error() {
        with_env(&[("STT_URLS", "x")], || {
            clear_all();
            unsafe {
                env::set_var("STT_URLS", "http://llama-swap:8080");
                env::set_var("TTS_URL", "http://kokoro:8080");
                env::set_var("PUBLIC_PORT", "not_a_number");
            }
            let err = Config::from_env().unwrap_err();
            assert!(matches!(
                err,
                ConfigError::InvalidValue {
                    key: "PUBLIC_PORT",
                    ..
                }
            ));
        });
    }
}
