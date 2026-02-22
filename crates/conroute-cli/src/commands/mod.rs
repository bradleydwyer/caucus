pub mod ask;
pub mod bench;
pub mod compare;
pub mod debate;
pub mod serve;

use conroute_core::{HttpProvider, LlmProvider, MultiProvider};

/// Default frontier model for each provider.
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.2";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-opus-4-6";
const DEFAULT_GEMINI_MODEL: &str = "gemini-3.1-pro-preview";

/// Return the default models based on which API keys are configured.
pub fn default_models() -> Vec<String> {
    let mut models = Vec::new();
    if std::env::var("OPENAI_API_KEY").is_ok() {
        models.push(DEFAULT_OPENAI_MODEL.to_string());
    }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        models.push(DEFAULT_ANTHROPIC_MODEL.to_string());
    }
    if std::env::var("GOOGLE_API_KEY").is_ok() {
        models.push(DEFAULT_GEMINI_MODEL.to_string());
    }
    if models.is_empty() {
        models.push("mock".to_string());
    }
    models
}

/// Build a MultiProvider from a list of model names, using environment variables for API keys.
pub fn build_provider(models: &[String]) -> anyhow::Result<MultiProvider> {
    let mut provider = MultiProvider::new();

    for model in models {
        let llm = build_single_provider(model)?;
        provider = provider.add(model.clone(), llm);
    }

    Ok(provider)
}

/// Build a single LLM provider from a model name.
pub fn build_single_provider(model: &str) -> anyhow::Result<Box<dyn LlmProvider>> {
    if model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3") {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set for model: {model}"))?;
        Ok(Box::new(HttpProvider::openai(key, model)))
    } else if model.starts_with("claude-") {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set for model: {model}"))?;
        Ok(Box::new(HttpProvider::anthropic(key, model)))
    } else if model.starts_with("gemini-") {
        let key = std::env::var("GOOGLE_API_KEY")
            .map_err(|_| anyhow::anyhow!("GOOGLE_API_KEY not set for model: {model}"))?;
        Ok(Box::new(HttpProvider::gemini(key, model)))
    } else if model == "mock" {
        Ok(Box::new(conroute_core::MockProvider::fixed(
            "This is a mock response for testing.",
        )))
    } else {
        // Default to OpenAI-compatible API
        let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let base_url =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
        Ok(Box::new(HttpProvider::new(base_url, key, model)))
    }
}
