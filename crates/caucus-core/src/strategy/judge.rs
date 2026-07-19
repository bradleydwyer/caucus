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
        Self { system_prompt: DEFAULT_JUDGE_SYSTEM.to_string(), rubric: None }
    }
}

pub(crate) const DEFAULT_JUDGE_SYSTEM: &str = "\
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
                let model_info =
                    c.model.as_ref().map(|m| format!(" (model: {m})")).unwrap_or_default();
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
        match parse_judge_response(&response) {
            Ok(parsed) => {
                // Out-of-range dissent indices are dropped, but recorded honestly.
                let (valid, dropped): (Vec<usize>, Vec<usize>) =
                    parsed.dissent_indices.iter().copied().partition(|&i| i < candidates.len());
                let dissents: Vec<Candidate> =
                    valid.iter().filter_map(|&i| candidates.get(i).cloned()).collect();

                let mut metadata = HashMap::new();
                metadata.insert("judge_parse".to_string(), serde_json::json!("structured"));
                if !dropped.is_empty() {
                    metadata.insert(
                        "warnings".to_string(),
                        serde_json::json!([format!(
                            "judge returned out-of-range dissent indices {dropped:?}; dropped"
                        )]),
                    );
                }

                Ok(ConsensusResult {
                    content: parsed.synthesis,
                    strategy: self.name().to_string(),
                    agreement_score: parsed.agreement_score,
                    candidates: candidates.to_vec(),
                    dissents,
                    reasoning: Some(parsed.reasoning),
                    metadata,
                })
            }
            Err(parse_err) => {
                // Fallback: use the raw response as the synthesis. Do NOT fabricate
                // an agreement score — we could not measure agreement, so report
                // the neutral midpoint and mark the result as unparsed.
                let mut metadata = HashMap::new();
                metadata.insert("judge_parse".to_string(), serde_json::json!("fallback"));
                metadata.insert(
                    "warnings".to_string(),
                    serde_json::json!([format!(
                        "judge response was not valid JSON ({parse_err}); \
                         agreement score is unknown and reported as 0.5"
                    )]),
                );
                Ok(ConsensusResult {
                    content: response,
                    strategy: self.name().to_string(),
                    agreement_score: 0.5,
                    candidates: candidates.to_vec(),
                    dissents: vec![],
                    reasoning: Some(
                        "Judge response was not in structured format; agreement unmeasured"
                            .to_string(),
                    ),
                    metadata,
                })
            }
        }
    }
}

#[derive(serde::Deserialize)]
pub(crate) struct JudgeResponse {
    pub(crate) synthesis: String,
    pub(crate) reasoning: String,
    pub(crate) agreement_score: f64,
    #[serde(default)]
    pub(crate) dissent_indices: Vec<usize>,
}

pub(crate) fn parse_judge_response(response: &str) -> Result<JudgeResponse> {
    fn normalize(mut parsed: JudgeResponse) -> JudgeResponse {
        // Clamp to the valid range, but never override the judge's reported
        // score based on dissent count — an empty dissent list does not imply
        // perfect agreement.
        parsed.agreement_score = parsed.agreement_score.clamp(0.0, 1.0);
        parsed
    }

    // Try direct parse first
    if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(response) {
        return Ok(normalize(parsed));
    }
    // Try to extract JSON from markdown code block
    if let Some(start) = response.find('{')
        && let Some(end) = response.rfind('}')
        && start <= end
    {
        let json_str = &response[start..=end];
        if let Ok(parsed) = serde_json::from_str::<JudgeResponse>(json_str) {
            return Ok(normalize(parsed));
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

    #[test]
    fn parse_keeps_reported_score_with_no_dissents() {
        // An empty dissent list must NOT inflate the agreement score to 1.0.
        let response = serde_json::json!({
            "synthesis": "s",
            "reasoning": "r",
            "agreement_score": 0.85,
            "dissent_indices": []
        });
        let parsed = parse_judge_response(&response.to_string()).unwrap();
        assert_eq!(parsed.agreement_score, 0.85);
        assert!(parsed.dissent_indices.is_empty());
    }

    #[test]
    fn parse_clamps_out_of_range_score() {
        let response = serde_json::json!({
            "synthesis": "s",
            "reasoning": "r",
            "agreement_score": 7.3,
            "dissent_indices": []
        });
        let parsed = parse_judge_response(&response.to_string()).unwrap();
        assert_eq!(parsed.agreement_score, 1.0);
    }

    #[test]
    fn parse_extracts_json_from_markdown_fence() {
        let response = "Here is my evaluation:\n```json\n{\"synthesis\":\"s\",\"reasoning\":\"r\",\"agreement_score\":0.4}\n```";
        let parsed = parse_judge_response(response).unwrap();
        assert_eq!(parsed.agreement_score, 0.4);
    }

    #[test]
    fn malformed_brace_order_returns_error_instead_of_panicking() {
        assert!(parse_judge_response("closing } before opening {").is_err());
    }

    #[tokio::test]
    async fn judge_fallback_does_not_fabricate_agreement() {
        let provider = MockProvider::fixed("This is not JSON at all.");
        let candidates = vec![Candidate::new("A"), Candidate::new("B")];

        let strategy = JudgeSynthesis::new();
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        // Unparseable judge output → neutral score, never a fabricated 1.0.
        assert_eq!(result.agreement_score, 0.5);
        assert_eq!(result.metadata["judge_parse"], serde_json::json!("fallback"));
        assert!(result.metadata["warnings"].as_array().unwrap().len() == 1);
    }

    #[tokio::test]
    async fn judge_drops_out_of_range_dissent_indices() {
        let judge_response = serde_json::json!({
            "synthesis": "s",
            "reasoning": "r",
            "agreement_score": 0.6,
            "dissent_indices": [1, 99]
        });
        let provider = MockProvider::fixed(judge_response.to_string());
        let candidates = vec![Candidate::new("A"), Candidate::new("B")];

        let strategy = JudgeSynthesis::new();
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.dissents.len(), 1);
        assert!(result.metadata.contains_key("warnings"));
    }
}
