use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::error::{ErrorKind, ProviderError};

/// A single response from an LLM (or any source) submitted for consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl Candidate {
    pub fn new(content: impl Into<String>) -> Self {
        Self { content: content.into(), model: None, confidence: None, metadata: HashMap::new() }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// The result of running a consensus strategy over a set of candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// The consensus output text.
    pub content: String,
    /// Which strategy produced this result.
    pub strategy: String,
    /// Agreement score from 0.0 (no agreement) to 1.0 (unanimous).
    pub agreement_score: f64,
    /// The original candidates that were evaluated.
    pub candidates: Vec<Candidate>,
    /// Candidates that dissented from the consensus.
    pub dissents: Vec<Candidate>,
    /// Explanation of how consensus was reached.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Additional metadata about the consensus process.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// The transport a provider uses to reach a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    /// Remote HTTP API (OpenAI-compatible, Anthropic, Gemini, ...).
    Api,
    /// A local CLI invoked with an explicit argv vector (never a shell).
    Command,
    /// A locally-running OpenAI-compatible server (Ollama, LM Studio, ...).
    LocalServer,
    /// Agent Client Protocol. Recognized but not supported by this build.
    Acp,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Api => "api",
            Self::Command => "command",
            Self::LocalServer => "local-server",
            Self::Acp => "acp",
        };
        f.write_str(s)
    }
}

/// Metadata about a single completion attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMeta {
    /// Model/provider name.
    pub provider: String,
    pub transport: Transport,
    pub latency_ms: u64,
    /// True when the output was truncated by a byte limit.
    pub truncated: bool,
    /// Error classification when the attempt failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorKind>,
}

/// The outcome of a single completion attempt: content plus normalized metadata.
#[derive(Debug)]
pub struct CompleteOutcome {
    pub result: Result<String>,
    pub meta: ResponseMeta,
}

/// Per-provider tunables for timeouts and output limits.
#[derive(Debug, Clone, Copy)]
pub struct ProviderOptions {
    /// Wall-clock timeout for one completion attempt.
    pub timeout: std::time::Duration,
    /// Maximum bytes of output to retain; excess is truncated.
    pub max_output_bytes: usize,
}

impl Default for ProviderOptions {
    fn default() -> Self {
        Self { timeout: std::time::Duration::from_secs(120), max_output_bytes: 1024 * 1024 }
    }
}

/// Trait for LLM providers. Users implement this to plug in any LLM backend.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a completion for the given prompt with an optional system message.
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String>;

    /// Generate embeddings for the given texts. Returns a vector of embeddings.
    /// Default implementation returns an error indicating embeddings aren't supported.
    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f64>>> {
        anyhow::bail!("Embedding not supported by this provider")
    }

    /// Which transport this provider uses.
    fn transport(&self) -> Transport {
        Transport::Api
    }

    /// Timeout and output-limit options for this provider.
    fn options(&self) -> ProviderOptions {
        ProviderOptions::default()
    }

    /// Complete and report normalized metadata (latency, transport, error kind).
    /// The default implementation wraps [`LlmProvider::complete`] with timing
    /// and error classification.
    async fn complete_meta(
        &self,
        name: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> CompleteOutcome {
        let start = std::time::Instant::now();
        let result = self.complete(prompt, system).await;
        let latency_ms = start.elapsed().as_millis() as u64;
        let error = result.as_ref().err().map(ProviderError::classify);
        CompleteOutcome {
            result,
            meta: ResponseMeta {
                provider: name.to_string(),
                transport: self.transport(),
                latency_ms,
                truncated: false,
                error,
            },
        }
    }
}

// Allow Box<dyn LlmProvider> to be used as an LlmProvider.
#[async_trait]
impl LlmProvider for Box<dyn LlmProvider> {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        self.as_ref().complete(prompt, system).await
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        self.as_ref().embed(texts).await
    }

    fn transport(&self) -> Transport {
        self.as_ref().transport()
    }

    fn options(&self) -> ProviderOptions {
        self.as_ref().options()
    }

    async fn complete_meta(
        &self,
        name: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> CompleteOutcome {
        self.as_ref().complete_meta(name, prompt, system).await
    }
}

// Allow Arc<dyn LlmProvider> to be used as an LlmProvider.
#[async_trait]
impl LlmProvider for std::sync::Arc<dyn LlmProvider> {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        self.as_ref().complete(prompt, system).await
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        self.as_ref().embed(texts).await
    }

    fn transport(&self) -> Transport {
        self.as_ref().transport()
    }

    fn options(&self) -> ProviderOptions {
        self.as_ref().options()
    }

    async fn complete_meta(
        &self,
        name: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> CompleteOutcome {
        self.as_ref().complete_meta(name, prompt, system).await
    }
}

/// Trait that all consensus strategies implement.
#[async_trait]
pub trait ConsensusStrategy: Send + Sync {
    /// The name of this strategy (e.g., "majority_vote", "judge_synthesis").
    fn name(&self) -> &str;

    /// Resolve consensus from the given candidates.
    /// Some strategies require an LLM provider (debate, judge, semantic clustering).
    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult>;

    /// Resolve with access to one provider per participant, enabling genuinely
    /// independent multi-provider behavior (e.g. blind debate where each
    /// participant refines its own position). The default implementation
    /// ignores `participants` and falls back to [`ConsensusStrategy::resolve`].
    async fn resolve_multi(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
        participants: Option<&crate::provider::MultiProvider>,
    ) -> Result<ConsensusResult> {
        let _ = participants;
        self.resolve(candidates, llm).await
    }
}
