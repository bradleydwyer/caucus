use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use crate::error::{ErrorKind, ProviderError, from_reqwest};
use crate::types::{LlmProvider, Transport};

/// Default per-request timeout for HTTP providers and provider fan-out.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// A mock LLM provider for testing. Returns canned responses.
pub struct MockProvider {
    responses: Vec<String>,
    index: std::sync::atomic::AtomicUsize,
}

impl MockProvider {
    pub fn new(responses: Vec<String>) -> Self {
        Self { responses, index: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// Create a mock that always returns the same response.
    pub fn fixed(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
        if self.responses.is_empty() {
            anyhow::bail!("mock provider has no configured responses");
        }
        let idx =
            self.index.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % self.responses.len();
        Ok(self.responses[idx].clone())
    }
}

/// An LLM provider backed by an OpenAI-compatible HTTP API.
pub struct HttpProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    timeout: Duration,
    transport: Transport,
}

impl HttpProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            timeout: DEFAULT_REQUEST_TIMEOUT,
            transport: Transport::Api,
        }
    }

    /// Create a provider for the OpenAI API.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1", api_key, model)
    }

    /// Create a provider for the Anthropic API.
    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.anthropic.com/v1", api_key, model)
    }

    /// Create a provider for the Google Gemini API.
    pub fn gemini(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://generativelanguage.googleapis.com", api_key, model)
    }

    /// Create a provider for the xAI (Grok) API.
    pub fn xai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.x.ai/v1", api_key, model)
    }

    /// Create a provider for a local Ollama server (no API key required).
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::local("http://localhost:11434/v1", model)
    }

    /// Create a provider for a local LM Studio server (no API key required).
    pub fn lmstudio(model: impl Into<String>) -> Self {
        Self::local("http://localhost:1234/v1", model)
    }

    /// Create a provider for a local OpenAI-compatible server (no API key).
    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let mut provider = Self::new(base_url, "", model);
        provider.transport = Transport::LocalServer;
        provider
    }

    /// Override the per-request timeout (default 120s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The model this provider is bound to.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a request with the configured timeout and normalize failures.
    async fn send(&self, req: reqwest::RequestBuilder) -> Result<serde_json::Value> {
        let attempt = async {
            let resp = req.send().await.map_err(|error| from_reqwest(&error))?;
            let resp = resp.error_for_status().map_err(|error| from_reqwest(&error))?;
            resp.json::<serde_json::Value>()
                .await
                .map_err(|error| ProviderError::new(ErrorKind::Parse, error.to_string()).into())
        };
        match tokio::time::timeout(self.timeout, attempt).await {
            Ok(result) => result,
            Err(_) => {
                Err(ProviderError::timeout(format!("exceeded {}s timeout", self.timeout.as_secs()))
                    .into())
            }
        }
    }
}

#[async_trait]
impl LlmProvider for HttpProvider {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        if self.base_url.contains("anthropic.com") {
            return self.complete_anthropic(prompt, system).await;
        }
        if self.base_url.contains("googleapis.com") {
            return self.complete_gemini(prompt, system).await;
        }
        self.complete_openai(prompt, system).await
    }

    fn transport(&self) -> Transport {
        self.transport
    }

    fn options(&self) -> crate::types::ProviderOptions {
        crate::types::ProviderOptions { timeout: self.timeout, ..Default::default() }
    }
}

impl HttpProvider {
    async fn complete_openai(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": prompt}));

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        let mut req = self.client.post(format!("{}/chat/completions", self.base_url)).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = self.send(req).await?;

        resp["choices"][0]["message"]["content"].as_str().map(str::to_string).ok_or_else(|| {
            ProviderError::new(ErrorKind::Parse, "Unexpected OpenAI response format").into()
        })
    }

    async fn complete_anthropic(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        });
        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }

        let req = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        let resp = self.send(req).await?;

        resp["content"][0]["text"].as_str().map(str::to_string).ok_or_else(|| {
            ProviderError::new(ErrorKind::Parse, "Unexpected Anthropic response format").into()
        })
    }

    async fn complete_gemini(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let mut contents = Vec::new();

        if let Some(sys) = system {
            contents.push(serde_json::json!({
                "role": "user",
                "parts": [{"text": sys}]
            }));
            contents.push(serde_json::json!({
                "role": "model",
                "parts": [{"text": "Understood."}]
            }));
        }
        contents.push(serde_json::json!({
            "role": "user",
            "parts": [{"text": prompt}]
        }));

        let body = serde_json::json!({
            "contents": contents,
        });

        let req = self
            .client
            .post(format!("{}/v1beta/models/{}:generateContent", self.base_url, self.model))
            .header("x-goog-api-key", &self.api_key)
            .json(&body);
        let resp = self.send(req).await?;

        resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                ProviderError::new(
                    ErrorKind::Parse,
                    format!("Unexpected Gemini response format: {resp}"),
                )
                .into()
            })
    }
}

/// A multi-model provider that dispatches to the right backend based on model name.
pub struct MultiProvider {
    providers: Vec<(String, Arc<dyn LlmProvider>)>,
}

impl MultiProvider {
    pub fn new() -> Self {
        Self { providers: Vec::new() }
    }

    /// Register a provider for a specific model name.
    pub fn add(mut self, model: impl Into<String>, provider: impl LlmProvider + 'static) -> Self {
        self.providers.push((model.into(), Arc::new(provider)));
        self
    }

    /// Register a pre-boxed provider (avoids double-boxing).
    pub fn add_shared(mut self, model: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        self.providers.push((model.into(), provider));
        self
    }

    /// Get the provider for a given model name.
    pub fn get(&self, model: &str) -> Option<&dyn LlmProvider> {
        self.providers
            .iter()
            .find(|(name, _)| name == model)
            .map(|(_, p)| p.as_ref() as &dyn LlmProvider)
    }

    /// Get a shared handle to the provider for a given model name.
    pub fn get_shared(&self, model: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers.iter().find(|(name, _)| name == model).map(|(_, p)| Arc::clone(p))
    }

    /// Iterate over (model, provider) pairs in registration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn LlmProvider>)> {
        self.providers.iter().map(|(name, p)| (name.as_str(), p))
    }

    /// List all registered model names.
    pub fn models(&self) -> Vec<&str> {
        self.providers.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether any providers are registered.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Generate completions from all registered models for the given prompt.
    ///
    /// Kept for backwards compatibility (sequential, no timeout). New code
    /// should prefer [`crate::fanout::bounded_fanout`] for bounded concurrency
    /// and normalized partial-failure reporting.
    pub async fn generate_all(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Vec<(String, Result<String>)> {
        let mut results = Vec::new();
        for (model, provider) in &self.providers {
            let result = provider.complete(prompt, system).await;
            results.push((model.clone(), result));
        }
        results
    }
}

impl Default for MultiProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Bounded, timeout-aware fan-out
// ---------------------------------------------------------------------------

/// Tunables for [`fanout`].
#[derive(Debug, Clone, Copy)]
pub struct FanoutConfig {
    /// Maximum number of in-flight requests.
    pub max_concurrency: usize,
    /// Hard timeout applied to each individual request.
    pub timeout: Duration,
    /// Minimum number of successful responses required.
    pub quorum: usize,
}

impl Default for FanoutConfig {
    fn default() -> Self {
        Self { max_concurrency: 4, timeout: DEFAULT_REQUEST_TIMEOUT, quorum: 1 }
    }
}

/// A successful fan-out response with provenance metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FanoutSuccess {
    pub model: String,
    pub transport: Transport,
    pub content: String,
    pub latency_ms: u64,
}

/// A failed fan-out response with a normalized error classification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FanoutFailure {
    pub model: String,
    pub transport: Transport,
    pub kind: ErrorKind,
    pub message: String,
    pub latency_ms: u64,
}

/// The aggregated result of a [`fanout`]: partial failures are data, not
/// exceptions. Use [`FanoutReport::quorum_met`] to decide whether the run
/// produced enough responses to proceed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FanoutReport {
    pub successes: Vec<FanoutSuccess>,
    pub failures: Vec<FanoutFailure>,
    pub quorum: usize,
}

impl FanoutReport {
    /// Whether at least `quorum` participants responded successfully.
    pub fn quorum_met(&self) -> bool {
        self.successes.len() >= self.quorum
    }

    /// A single-line warning per failure, for receipts and stderr output.
    pub fn warnings(&self) -> Vec<String> {
        self.failures
            .iter()
            .map(|f| format!("{} failed ({}): {}", f.model, f.kind, f.message))
            .collect()
    }
}

/// Query every provider in `multi` for the same prompt with bounded
/// concurrency and a per-request timeout.
///
/// Partial failure is normal: every participant's outcome (success with
/// latency/transport metadata, or a classified failure) is recorded in the
/// returned report. The caller decides whether [`FanoutReport::quorum_met`]
/// is sufficient to continue.
pub async fn fanout(
    multi: &MultiProvider,
    prompt: &str,
    system: Option<&str>,
    config: FanoutConfig,
) -> FanoutReport {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrency.max(1)));
    let mut join_set = tokio::task::JoinSet::new();

    for (model, provider) in multi.iter() {
        let permit = Arc::clone(&semaphore);
        let provider = Arc::clone(provider);
        let model = model.to_string();
        let prompt = prompt.to_string();
        let system = system.map(str::to_string);
        let timeout = config.timeout;

        join_set.spawn(async move {
            let _permit = permit.acquire().await.expect("semaphore not closed");
            let started = std::time::Instant::now();
            let transport = provider.transport();
            let result =
                tokio::time::timeout(timeout, provider.complete(&prompt, system.as_deref())).await;
            let latency_ms = started.elapsed().as_millis() as u64;

            match result {
                Ok(Ok(content)) => Ok(FanoutSuccess { model, transport, content, latency_ms }),
                Ok(Err(e)) => Err(FanoutFailure {
                    model,
                    transport,
                    kind: ProviderError::classify(&e),
                    message: e.to_string(),
                    latency_ms,
                }),
                Err(_) => Err(FanoutFailure {
                    model,
                    transport,
                    kind: ErrorKind::Timeout,
                    message: format!("exceeded {}s timeout", timeout.as_secs()),
                    latency_ms,
                }),
            }
        });
    }

    let mut successes = Vec::new();
    let mut failures = Vec::new();
    while let Some(outcome) = join_set.join_next().await {
        match outcome {
            Ok(Ok(success)) => successes.push(success),
            Ok(Err(failure)) => failures.push(failure),
            Err(join_err) => failures.push(FanoutFailure {
                model: "unknown".into(),
                transport: Transport::Api,
                kind: ErrorKind::Other,
                message: format!("task panicked: {join_err}"),
                latency_ms: 0,
            }),
        }
    }

    // Deterministic output: restore registration order.
    let order: Vec<&str> = multi.models();
    let rank = |model: &str| order.iter().position(|m| *m == model).unwrap_or(usize::MAX);
    successes.sort_by_key(|s| rank(&s.model));
    failures.sort_by_key(|f| rank(&f.model));

    FanoutReport { successes, failures, quorum: config.quorum }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    struct FailProvider;

    #[async_trait]
    impl LlmProvider for FailProvider {
        async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
            Err(ProviderError::new(ErrorKind::Unavailable, "always down").into())
        }
    }

    struct SlowProvider;

    #[async_trait]
    impl LlmProvider for SlowProvider {
        async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok("too late".into())
        }
    }

    #[test]
    fn local_constructors_use_local_server_transport() {
        assert_eq!(HttpProvider::ollama("llama3.2").transport(), Transport::LocalServer);
        assert_eq!(HttpProvider::lmstudio("qwen").transport(), Transport::LocalServer);
        assert_eq!(HttpProvider::openai("k", "gpt").transport(), Transport::Api);
    }

    #[tokio::test]
    async fn empty_mock_provider_returns_error_instead_of_panicking() {
        let error = MockProvider::new(vec![]).complete("q", None).await.unwrap_err();
        assert!(error.to_string().contains("no configured responses"));
    }

    #[tokio::test]
    async fn http_timeout_covers_a_stalled_response_body() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n{",
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let provider = HttpProvider::local(format!("http://{address}"), "test-model")
            .with_timeout(Duration::from_millis(75));
        let error = provider.complete("hello", None).await.unwrap_err();
        assert_eq!(ProviderError::classify(&error), ErrorKind::Timeout);
        server.abort();
    }

    #[tokio::test]
    async fn fanout_collects_successes_and_failures() {
        let multi = MultiProvider::new()
            .add("good", MockProvider::fixed("answer"))
            .add("bad", FailProvider);

        let report =
            fanout(&multi, "q", None, FanoutConfig { quorum: 1, ..Default::default() }).await;

        assert_eq!(report.successes.len(), 1);
        assert_eq!(report.successes[0].model, "good");
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].model, "bad");
        assert_eq!(report.failures[0].kind, ErrorKind::Unavailable);
        assert!(report.quorum_met());
        assert_eq!(report.warnings().len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn fanout_enforces_timeout() {
        let multi = MultiProvider::new().add("slow", SlowProvider);

        let config =
            FanoutConfig { timeout: Duration::from_millis(50), quorum: 1, ..Default::default() };
        let report = fanout(&multi, "q", None, config).await;

        assert!(report.successes.is_empty());
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].kind, ErrorKind::Timeout);
        assert!(!report.quorum_met());
    }

    #[tokio::test]
    async fn fanout_quorum_not_met_when_too_many_fail() {
        let multi = MultiProvider::new()
            .add("ok", MockProvider::fixed("yes"))
            .add("bad1", FailProvider)
            .add("bad2", FailProvider);

        let report =
            fanout(&multi, "q", None, FanoutConfig { quorum: 2, ..Default::default() }).await;

        assert_eq!(report.successes.len(), 1);
        assert!(!report.quorum_met());
    }

    #[tokio::test]
    async fn fanout_preserves_registration_order() {
        let multi = MultiProvider::new()
            .add("c", MockProvider::fixed("3"))
            .add("a", MockProvider::fixed("1"))
            .add("b", MockProvider::fixed("2"));

        let report = fanout(&multi, "q", None, FanoutConfig::default()).await;
        let models: Vec<&str> = report.successes.iter().map(|s| s.model.as_str()).collect();
        assert_eq!(models, ["c", "a", "b"]);
    }
}
