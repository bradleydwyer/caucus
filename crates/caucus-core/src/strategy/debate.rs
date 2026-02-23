use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::strategy::judge::parse_judge_response;
use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// Multi-round debate where candidates see each other's responses and refine their positions.
/// Includes convergence detection to stop early when positions stabilize.
pub struct MultiRoundDebate {
    pub config: DebateConfig,
}

pub struct DebateConfig {
    /// Maximum number of debate rounds.
    pub max_rounds: usize,
    /// Convergence threshold — if positions change less than this between rounds, stop early.
    pub convergence_threshold: f64,
    /// System prompt for debate participants.
    pub system_prompt: String,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            convergence_threshold: 0.9,
            system_prompt: DEFAULT_DEBATE_SYSTEM.to_string(),
        }
    }
}

const DEFAULT_DEBATE_SYSTEM: &str = "\
You are participating in a structured debate with other AI models. \
You have been given a question and will see other participants' responses. \
Carefully consider their arguments. If they make valid points, incorporate them. \
If you disagree, explain why with clear reasoning. \
Your goal is to arrive at the most accurate and well-reasoned answer.";

const DEBATE_ROUND_PROMPT: &str = "\
Original question: {question}

Here are the current positions from all participants:

{positions}

This is round {round} of {max_rounds}. \
Review the other positions and refine your answer. \
If you've changed your mind on any points, explain why. \
If you maintain your position, strengthen your argument.";

const DEBATE_JUDGE_PROMPT: &str = "\
Below is a synthesis produced by a multi-round debate, followed by the {count} original \
candidate responses that entered the debate.

--- Debate Synthesis ---
{synthesis}

{candidates}

Compare the synthesis to each original candidate. Determine how much the candidates \
agree with the final synthesis overall, and identify any candidates whose position \
significantly differs from the consensus.

Respond in the following JSON format:
{{
  \"synthesis\": \"(copy the debate synthesis above verbatim)\",
  \"reasoning\": \"Brief explanation of agreement and disagreements\",
  \"agreement_score\": 0.0 to 1.0 representing overall agreement,
  \"dissent_indices\": [zero-based indices of candidates that significantly disagreed]
}}";

impl MultiRoundDebate {
    pub fn new() -> Self {
        Self { config: DebateConfig::default() }
    }

    pub fn with_config(config: DebateConfig) -> Self {
        Self { config }
    }

    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.config.max_rounds = rounds;
        self
    }

    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }
}

impl Default for MultiRoundDebate {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute simple similarity between two strings for convergence detection.
fn text_similarity(a: &str, b: &str) -> f64 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    if a == b {
        return 1.0;
    }

    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

#[async_trait]
impl ConsensusStrategy for MultiRoundDebate {
    fn name(&self) -> &str {
        "multi_round_debate"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let llm =
            llm.ok_or_else(|| anyhow::anyhow!("MultiRoundDebate requires an LLM provider"))?;

        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }

        let mut current_positions: Vec<String> =
            candidates.iter().map(|c| c.content.clone()).collect();
        let mut round_history: Vec<Vec<String>> = vec![current_positions.clone()];

        // Extract the original question from metadata if available,
        // otherwise use a generic prompt
        let question = candidates
            .first()
            .and_then(|c| c.metadata.get("question"))
            .and_then(|v| v.as_str())
            .unwrap_or("(see the responses below)");

        let mut actual_rounds = 0;
        for round in 1..=self.config.max_rounds {
            actual_rounds = round;

            let positions_text = current_positions
                .iter()
                .enumerate()
                .map(|(i, pos)| {
                    let model =
                        candidates.get(i).and_then(|c| c.model.as_deref()).unwrap_or("Participant");
                    format!("--- {} (Position {}) ---\n{}", model, i + 1, pos)
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            let prompt = DEBATE_ROUND_PROMPT
                .replace("{question}", question)
                .replace("{positions}", &positions_text)
                .replace("{round}", &round.to_string())
                .replace("{max_rounds}", &self.config.max_rounds.to_string());

            // Have the LLM produce a refined position
            let refined = llm.complete(&prompt, Some(&self.config.system_prompt)).await?;

            // Check convergence: compare refined with each current position
            let max_similarity = current_positions
                .iter()
                .map(|pos| text_similarity(&refined, pos))
                .fold(0.0_f64, f64::max);

            // Update all positions to the refined one (simplified: single-LLM debate)
            let new_positions = vec![refined; current_positions.len()];
            current_positions = new_positions;
            round_history.push(current_positions.clone());

            if max_similarity >= self.config.convergence_threshold {
                tracing::info!(
                    "Debate converged at round {} (similarity: {:.2})",
                    round,
                    max_similarity
                );
                break;
            }
        }

        // The final position is the consensus
        let final_position = current_positions.into_iter().next().unwrap_or_default();

        // Use an LLM judge to classify agreement/dissent, falling back to
        // text_similarity if the judge response can't be parsed.
        let candidates_text = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let model_info =
                    c.model.as_ref().map(|m| format!(" (model: {m})")).unwrap_or_default();
                format!("--- Candidate {}{}---\n{}", i, model_info, c.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let judge_prompt = DEBATE_JUDGE_PROMPT
            .replace("{count}", &candidates.len().to_string())
            .replace("{synthesis}", &final_position)
            .replace("{candidates}", &candidates_text);

        let judge_response = llm.complete(&judge_prompt, Some(&self.config.system_prompt)).await?;

        let (avg_agreement, dissents) = if let Ok(parsed) = parse_judge_response(&judge_response) {
            let dissents: Vec<Candidate> =
                parsed.dissent_indices.iter().filter_map(|&i| candidates.get(i).cloned()).collect();
            (parsed.agreement_score, dissents)
        } else {
            // Fallback: text_similarity scoring
            let agreement_scores: Vec<f64> =
                candidates.iter().map(|c| text_similarity(&final_position, &c.content)).collect();
            let avg = agreement_scores.iter().sum::<f64>() / agreement_scores.len().max(1) as f64;
            let dissents: Vec<Candidate> = candidates
                .iter()
                .zip(agreement_scores.iter())
                .filter(|&(_, &score)| score < 0.3)
                .map(|(c, _)| c.clone())
                .collect();
            (avg, dissents)
        };

        let mut metadata = HashMap::new();
        metadata.insert("rounds_completed".to_string(), serde_json::json!(actual_rounds));
        metadata.insert("round_history".to_string(), serde_json::json!(round_history));

        Ok(ConsensusResult {
            content: final_position,
            strategy: self.name().to_string(),
            agreement_score: avg_agreement,
            candidates: candidates.to_vec(),
            dissents,
            reasoning: Some(format!(
                "Debate completed in {} round(s) of {} maximum",
                actual_rounds, self.config.max_rounds,
            )),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[tokio::test]
    async fn debate_converges() {
        let judge_json = serde_json::json!({
            "synthesis": "The refined consensus answer after debate.",
            "reasoning": "Both candidates broadly agreed.",
            "agreement_score": 0.85,
            "dissent_indices": []
        });

        let provider = MockProvider::new(vec![
            "The refined consensus answer after debate.".to_string(),
            judge_json.to_string(),
        ]);
        let candidates = vec![
            Candidate::new("Answer A from model 1").with_model("model-1"),
            Candidate::new("Answer B from model 2").with_model("model-2"),
        ];

        let strategy = MultiRoundDebate::new().with_rounds(3);
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.strategy, "multi_round_debate");
        assert!(!result.content.is_empty());
        assert_eq!(result.agreement_score, 0.85);
        assert!(result.dissents.is_empty());
    }

    #[tokio::test]
    async fn debate_requires_llm() {
        let candidates = vec![Candidate::new("test")];
        let strategy = MultiRoundDebate::new();
        let result = strategy.resolve(&candidates, None).await;
        assert!(result.is_err());
    }
}
