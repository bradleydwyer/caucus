//! Normalized provider error classification.
//!
//! [`ProviderError`] pairs a machine-readable [`ErrorKind`] with a message so
//! callers (fan-out reports, receipts, CLI warnings) can distinguish timeouts,
//! auth failures, and parse errors without string matching.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Classification of a provider failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorKind {
    /// Request exceeded its wall-clock timeout.
    Timeout,
    /// Authentication or authorization failed (HTTP 401/403).
    Auth,
    /// Rate limited by the upstream API (HTTP 429).
    RateLimited,
    /// Connection failed or the upstream service is unavailable (5xx).
    Unavailable,
    /// The response could not be parsed into the expected shape.
    Parse,
    /// The capability is recognized but deliberately not supported by this build.
    Unsupported,
    /// Anything not covered above.
    Other,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Timeout => "timeout",
            Self::Auth => "auth",
            Self::RateLimited => "rate-limited",
            Self::Unavailable => "unavailable",
            Self::Parse => "parse",
            Self::Unsupported => "unsupported",
            Self::Other => "other",
        };
        f.write_str(s)
    }
}

/// An error raised by a provider, carrying a normalized [`ErrorKind`].
#[derive(Debug)]
pub struct ProviderError {
    kind: ErrorKind,
    message: String,
}

impl ProviderError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    /// Construct a timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }

    /// The normalized classification of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Classify any error: returns the [`ErrorKind`] of a [`ProviderError`],
    /// or [`ErrorKind::Other`] for anything else.
    pub fn classify(err: &anyhow::Error) -> ErrorKind {
        err.downcast_ref::<ProviderError>().map(|e| e.kind).unwrap_or(ErrorKind::Other)
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ProviderError {}

/// Classify a [`reqwest`] failure into a [`ProviderError`].
pub fn from_reqwest(err: &reqwest::Error) -> ProviderError {
    let kind = if err.is_timeout() {
        ErrorKind::Timeout
    } else if err.is_connect() {
        ErrorKind::Unavailable
    } else if let Some(status) = err.status() {
        match status.as_u16() {
            401 | 403 => ErrorKind::Auth,
            429 => ErrorKind::RateLimited,
            s if s >= 500 => ErrorKind::Unavailable,
            _ => ErrorKind::Other,
        }
    } else {
        ErrorKind::Other
    };
    ProviderError::new(kind, err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_preserves_provider_error_kind() {
        let err: anyhow::Error = ProviderError::new(ErrorKind::Auth, "bad key").into();
        assert_eq!(ProviderError::classify(&err), ErrorKind::Auth);
    }

    #[test]
    fn classify_falls_back_to_other() {
        let err = anyhow::anyhow!("boom");
        assert_eq!(ProviderError::classify(&err), ErrorKind::Other);
    }

    #[test]
    fn timeout_constructor_sets_kind() {
        let err = ProviderError::timeout("too slow");
        assert_eq!(err.kind(), ErrorKind::Timeout);
        assert_eq!(err.to_string(), "too slow");
    }
}
