use anyhow::Result;
use async_trait::async_trait;

use crate::types::LlmProvider;

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

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Unexpected OpenAI response format"))?;

        Ok(content.to_string())
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

        let resp = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;

        let content = resp["content"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Unexpected Anthropic response format"))?;

        Ok(content.to_string())
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

        let resp = self
            .client
            .post(format!("{}/v1beta/models/{}:generateContent", self.base_url, self.model))
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;

        let content = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Unexpected Gemini response format: {resp}"))?;

        Ok(content.to_string())
    }
}

/// A multi-model provider that dispatches to the right backend based on model name.
pub struct MultiProvider {
    providers: Vec<(String, Box<dyn LlmProvider>)>,
}

impl MultiProvider {
    pub fn new() -> Self {
        Self { providers: Vec::new() }
    }

    /// Register a provider for a specific model name.
    pub fn add(mut self, model: impl Into<String>, provider: impl LlmProvider + 'static) -> Self {
        self.providers.push((model.into(), Box::new(provider)));
        self
    }

    /// Get the provider for a given model name.
    pub fn get(&self, model: &str) -> Option<&dyn LlmProvider> {
        self.providers.iter().find(|(name, _)| name == model).map(|(_, p)| p.as_ref())
    }

    /// List all registered model names.
    pub fn models(&self) -> Vec<&str> {
        self.providers.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Generate completions from all registered models for the given prompt.
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
