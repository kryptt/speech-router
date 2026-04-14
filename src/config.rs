use std::env;
use std::fmt;
use std::net::SocketAddr;

/// Validated, immutable configuration.
/// All parsing happens in `from_env`; once constructed, every field is trusted.
#[derive(Debug, Clone)]
pub struct Config {
    pub speaches_url: String,
    pub public_addr: SocketAddr,
    pub wyoming_port: u16,
    pub internal_addr: SocketAddr,
    pub default_model: String,
    pub default_tts_model: String,
    pub default_tts_voice: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let speaches_url =
            env::var("SPEACHES_URL").map_err(|_| ConfigError::MissingRequired("SPEACHES_URL"))?;

        if speaches_url.is_empty() {
            return Err(ConfigError::MissingRequired("SPEACHES_URL"));
        }

        let speaches_url = speaches_url.trim_end_matches('/').to_string();

        let public_port = parse_env_u16("PUBLIC_PORT", 8000)?;
        let wyoming_port = parse_env_u16("WYOMING_PORT", 10300)?;
        let internal_port = parse_env_u16("INTERNAL_PORT", 9090)?;

        let default_model =
            env::var("DEFAULT_MODEL").unwrap_or_else(|_| "large-v3-turbo".to_string());
        let default_tts_model =
            env::var("DEFAULT_TTS_MODEL").unwrap_or_else(|_| "kokoro".to_string());
        let default_tts_voice =
            env::var("DEFAULT_TTS_VOICE").unwrap_or_else(|_| "af_heart".to_string());

        Ok(Config {
            speaches_url,
            public_addr: SocketAddr::from(([0, 0, 0, 0], public_port)),
            wyoming_port,
            internal_addr: SocketAddr::from(([0, 0, 0, 0], internal_port)),
            default_model,
            default_tts_model,
            default_tts_voice,
        })
    }
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

    #[test]
    fn from_env_with_defaults() {
        with_env(&[("SPEACHES_URL", "http://speaches:8000")], || {
            // Clear optional vars to ensure defaults
            for key in [
                "PUBLIC_PORT",
                "WYOMING_PORT",
                "INTERNAL_PORT",
                "DEFAULT_MODEL",
                "DEFAULT_TTS_MODEL",
                "DEFAULT_TTS_VOICE",
            ] {
                // SAFETY: tests run single-threaded; no concurrent env readers.
                unsafe { env::remove_var(key) };
            }

            let config = Config::from_env().expect("should parse with defaults");
            assert_eq!(config.speaches_url, "http://speaches:8000");
            assert_eq!(config.public_addr.port(), 8000);
            assert_eq!(config.wyoming_port, 10300);
            assert_eq!(config.internal_addr.port(), 9090);
            assert_eq!(config.default_model, "large-v3-turbo");
            assert_eq!(config.default_tts_model, "kokoro");
            assert_eq!(config.default_tts_voice, "af_heart");
        });
    }

    #[test]
    fn from_env_with_all_vars() {
        with_env(
            &[
                ("SPEACHES_URL", "http://custom:9999/"),
                ("PUBLIC_PORT", "7000"),
                ("WYOMING_PORT", "7001"),
                ("INTERNAL_PORT", "7002"),
                ("DEFAULT_MODEL", "tiny"),
                ("DEFAULT_TTS_MODEL", "piper"),
                ("DEFAULT_TTS_VOICE", "en_US"),
            ],
            || {
                let config = Config::from_env().expect("should parse all vars");
                assert_eq!(config.speaches_url, "http://custom:9999");
                assert_eq!(config.public_addr.port(), 7000);
                assert_eq!(config.wyoming_port, 7001);
                assert_eq!(config.internal_addr.port(), 7002);
                assert_eq!(config.default_model, "tiny");
                assert_eq!(config.default_tts_model, "piper");
                assert_eq!(config.default_tts_voice, "en_US");
            },
        );
    }

    #[test]
    fn missing_speaches_url_is_error() {
        // Use with_env to set a dummy, then remove inside the closure.
        // with_env restores the original value on exit.
        with_env(&[("SPEACHES_URL", "__placeholder__")], || {
            // SAFETY: tests run single-threaded; no concurrent env readers.
            unsafe { env::remove_var("SPEACHES_URL") };
            let err = Config::from_env().unwrap_err();
            assert!(matches!(err, ConfigError::MissingRequired("SPEACHES_URL")));
        });
    }

    #[test]
    fn invalid_port_is_error() {
        with_env(
            &[
                ("SPEACHES_URL", "http://x:1"),
                ("PUBLIC_PORT", "not_a_number"),
            ],
            || {
                let err = Config::from_env().unwrap_err();
                assert!(matches!(
                    err,
                    ConfigError::InvalidValue {
                        key: "PUBLIC_PORT",
                        ..
                    }
                ));
            },
        );
    }
}
