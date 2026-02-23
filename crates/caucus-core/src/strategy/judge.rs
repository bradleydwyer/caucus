use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// A consensus strategy that uses a separate LLM as a judge to evaluate
/// all candidates and synthesize the best response.
pub struct JudgeSynthesis {
    /// System prompt for the judge LLM.
    pub system_prompt: String,
    /// Rubric/criteria for evaluation.
    pub rubric: Option<String>,
}

impl Default for JudgeSynthesis {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_JUDGE_SYSTEM.to_string(),
            rubric: None,
        }
    }
}

const DEFAULT_JUDGE_SYSTEM: &str = "\
You are an expert judge evaluating multiple AI responses to the same question. \
Your job is to synthesize the best possible answer by analyzing all responses, \
identifying the strongest reasoning and most accurate information from each, \
and producing a single authoritative response.";

const DEFAULT_JUDGE_PROMPT: &str = "\
Below are {count} responses to the same question. Evaluate each response for accuracy, \
completeness, and reasoning quality. Then synthesize the best possible answer.

{candidates}

Respond in the following JSON format:
{{
  \"synthesis\": \"Your synthesized best answer\",
  \"reasoning\": \"Brief explanation of how you evaluated and combined the responses\",
  \"agreement_score\": 0.0 to 1.0 representing how much the responses agreed,
  \"dissent_indices\": [indices of responses that significantly disagreed with the consensus]
}}";

impl JudgeSynthesis {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = Some(rubric.into());
        self
    }

    fn build_prompt(&self, candidates: &[Candidate]) -> String {
        let candidates_text = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let model_info = c
                    .model
                    .as_ref()
                    .map(|m| format!(" (model: {m})"))
                    .unwrap_or_default();
                format!("--- Response {}{}---\n{}", i + 1, model_info, c.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut prompt = DEFAULT_JUDGE_PROMPT
            .replace("{count}", &candidates.len().to_string())
            .replace("{candidates}", &candidates_text);

        if let Some(rubric) = &self.rubric {
            prompt = format!("Evaluation rubric: {rubric}\n\n{prompt}");
        }

        prompt
    }
}

#[async_trait]
impl ConsensusStrategy for JudgeSynthesis {
    fn name(&self) -> &str {
        "judge_synthesis"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let llm = llm.ok_or_else(|| anyhow::anyhow!("JudgeSynthesis requires an LLM provider"))?;

        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }

        let prompt = self.build_prompt(candidates);
        let response = llm.complete(&prompt, Some(&self.system_prompt)).await?;

        // Try to parse structured JSON response
        if let Ok(parsed) = parse_judge_response(&response) {
            let dissents: Vec<Candidate> = parsed
                .dissent_indices
                .iter()
                .filter_map(|&i| candidates.get(i).cloned())
                .collect();

            Ok(ConsensusResult {
                content: parsed.synthesis,
                strategy: self.name().to_string(),
                agreement_score: parsed.agreement_score,
                candidates: candidates.to_vec(),
                dissents,
                reasoning: Some(parsed.reasoning),
                metadata: HashMap::new(),
            })
        } else {
            // Fallback: use the raw response as the synthesis
            Ok(ConsensusResult {
                content: response,
                strategy: self.name().to_string(),
                agreement_score: 0.5,
                candidates: candidates.to_vec(),
                dissents: vec![],
                reasoning: Some("Judge response was not in structured format".to_string()),
                metadata: HashMap::new(),
            })
        }
    }
}

#[derive(serde::Deserialize)]
struct JudgeResponse {
    synthesis: String,
    reasoning: String,
    agreement_score: f64,
    #[serde(default)]
    dissent_indices: Vec<usize>,
}

fn parse_judge_response(response: &str) -> Result<JudgeResponse> {
    // Try direct parse first
    if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(response) {
        return Ok(parsed);
    }
    // Try to extract JSON from markdown code block
    if let Some(start) = response.find('{')
        && let Some(end) = response.rfind('}')
    {
        let json_str = &response[start..=end];
        if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(json_str) {
            return Ok(parsed);
        }
    }
    anyhow::bail!("Could not parse judge response as JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[tokio::test]
    async fn judge_synthesis_basic() {
        let judge_response = serde_json::json!({
            "synthesis": "The synthesized answer combining the best parts.",
            "reasoning": "Response 1 had better reasoning, Response 2 had more detail.",
            "agreement_score": 0.7,
            "dissent_indices": [1]
        });

        let provider = MockProvider::fixed(judge_response.to_string());
        let candidates = vec![
            Candidate::new("Answer from model A"),
            Candidate::new("Different answer from model B"),
            Candidate::new("Similar to model A's answer"),
        ];

        let strategy = JudgeSynthesis::new();
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.content, "The synthesized answer combining the best parts.");
        assert_eq!(result.agreement_score, 0.7);
        assert_eq!(result.dissents.len(), 1);
    }

    #[tokio::test]
    async fn judge_requires_llm() {
        let candidates = vec![Candidate::new("test")];
        let strategy = JudgeSynthesis::new();
        let result = strategy.resolve(&candidates, None).await;
        assert!(result.is_err());
    }
}
