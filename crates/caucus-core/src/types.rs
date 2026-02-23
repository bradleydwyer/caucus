use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
}
