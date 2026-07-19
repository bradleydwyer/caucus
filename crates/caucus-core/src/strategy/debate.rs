use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use futures::future::join_all;

use crate::provider::MultiProvider;
use crate::strategy::judge::{DEFAULT_JUDGE_SYSTEM, parse_judge_response};
use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// Multi-round debate where each participant independently blind-critiques the
/// current positions and revises its own. Includes convergence detection to
/// stop early when positions stabilize.
///
/// "Blind" means positions are anonymized (no model names) and re-shuffled for
/// every participant in every round, so no participant can tell which position
/// is its own or anyone else's. "Independent" means each participant produces
/// its own revised position via its own provider — there is no single
/// broadcast rewrite shared by all.
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
You are participating in a structured deliberation with other AI models. \
You will see anonymized positions; you do not know which participant wrote which \
(one of them is your own earlier answer). \
Critique each position on its merits. If a position makes valid points, incorporate them. \
If it is wrong, explain why with clear reasoning. \
Your goal is to arrive at the most accurate and well-reasoned answer.";

const BLIND_CRITIQUE_PROMPT: &str = "\
Original question: {question}

Below are the current positions in an ongoing deliberation. They are anonymized \
and shuffled; you do not know which participant wrote which position — one of \
them is your own earlier answer.

{positions}

This is round {round} of {max_rounds}.

First, briefly identify the strongest weakness or gap in EACH position. \
Then write your own revised, standalone answer to the original question, \
incorporating whatever survives your critique. Do NOT reference the debate \
process, position numbers, or \"other participants\" in the final answer — \
write as if you are the sole author giving a definitive response.

End your response with a line containing only \"FINAL ANSWER:\" followed by \
your final answer.";

const DEBATE_JUDGE_PROMPT: &str = "\
Below are the {count} final positions produced by an independent multi-round debate \
on the same question.

{candidates}

Evaluate how much the positions agree overall, synthesize the best possible answer \
from them, and identify any positions that significantly diverge from the consensus.

Respond in the following JSON format:
{{
  \"synthesis\": \"Your synthesized best answer\",
  \"reasoning\": \"Brief explanation of agreement and disagreements\",
  \"agreement_score\": 0.0 to 1.0 representing overall agreement,
  \"dissent_indices\": [zero-based indices of positions that significantly disagreed]
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

    /// Run the debate with an explicit provider per participant.
    ///
    /// `participant_llms[i]` refines `candidates[i]`'s position. `judge`
    /// adjudicates the final positions.
    pub async fn resolve_with_participants(
        &self,
        candidates: &[Candidate],
        participant_llms: &[&dyn LlmProvider],
        judge: &dyn LlmProvider,
    ) -> Result<ConsensusResult> {
        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }
        if participant_llms.len() != candidates.len() {
            anyhow::bail!(
                "expected {} participant providers, got {}",
                candidates.len(),
                participant_llms.len()
            );
        }

        // Extract the original question from metadata if available,
        // otherwise use a generic prompt.
        let question = candidates
            .first()
            .and_then(|c| c.metadata.get("question"))
            .and_then(|v| v.as_str())
            .unwrap_or("(see the responses below)");

        let mut current_positions: Vec<String> =
            candidates.iter().map(|c| c.content.clone()).collect();
        let mut round_history: Vec<Vec<String>> = vec![current_positions.clone()];
        let mut warnings = Vec::new();

        let mut actual_rounds = 0;
        for round in 1..=self.config.max_rounds {
            actual_rounds = round;

            // Each participant independently blind-critiques the shuffled,
            // anonymized positions and revises only its own. `join_all` polls
            // borrowed provider futures concurrently without requiring the
            // `'static` lifetime that spawned tasks would need.
            let attempts = participant_llms.iter().enumerate().map(|(i, llm)| {
                let prompt = blind_critique_prompt(
                    question,
                    &current_positions,
                    i,
                    round,
                    self.config.max_rounds,
                );
                async move {
                    let result = tokio::time::timeout(
                        llm.options().timeout,
                        llm.complete(&prompt, Some(&self.config.system_prompt)),
                    )
                    .await;
                    (i, result)
                }
            });
            let mut new_positions = Vec::with_capacity(current_positions.len());
            let mut refined_indices = Vec::with_capacity(current_positions.len());
            for (i, result) in join_all(attempts).await {
                match result {
                    Ok(Ok(refined)) => {
                        let refined = extract_final_answer(&refined);
                        if refined.trim().is_empty() {
                            warnings.push(format!(
                                "debate round {round}: participant {i} returned an empty refinement; retained its previous position"
                            ));
                            new_positions.push(current_positions[i].clone());
                        } else {
                            refined_indices.push(i);
                            new_positions.push(refined);
                        }
                    }
                    Ok(Err(error)) => {
                        warnings.push(format!(
                            "debate round {round}: participant {i} failed ({error}); retained its previous position"
                        ));
                        new_positions.push(current_positions[i].clone());
                    }
                    Err(_) => {
                        warnings.push(format!(
                            "debate round {round}: participant {i} exceeded its request timeout; retained its previous position"
                        ));
                        new_positions.push(current_positions[i].clone());
                    }
                }
            }

            // Failed/empty turns retain their old positions for resilience,
            // but must not count as perfect similarity and fabricate
            // convergence. Only successful refinements contribute.
            let similarities: Vec<f64> = refined_indices
                .iter()
                .map(|&i| text_similarity(&current_positions[i], &new_positions[i]))
                .collect();
            let mean_similarity = (!similarities.is_empty())
                .then(|| similarities.iter().sum::<f64>() / similarities.len() as f64);

            current_positions = new_positions;
            round_history.push(current_positions.clone());

            if let Some(mean_similarity) = mean_similarity {
                tracing::info!(
                    "Debate round {}/{} complete (mean successful-position similarity: {:.3})",
                    round,
                    self.config.max_rounds,
                    mean_similarity
                );
                if mean_similarity >= self.config.convergence_threshold {
                    tracing::info!(
                        "Debate converged early at round {} (threshold: {:.2})",
                        round,
                        self.config.convergence_threshold
                    );
                    break;
                }
            } else {
                tracing::warn!(
                    "Debate round {}/{} had no successful refinements; convergence not evaluated",
                    round,
                    self.config.max_rounds
                );
            }
        }

        self.adjudicate(
            candidates,
            &current_positions,
            judge,
            actual_rounds,
            round_history,
            warnings,
        )
        .await
    }

    /// Adjudicate the final positions with a judge LLM, falling back to
    /// deterministic text-similarity scoring when the judge output cannot be
    /// parsed. The fallback never fabricates a perfect agreement score.
    async fn adjudicate(
        &self,
        candidates: &[Candidate],
        final_positions: &[String],
        judge: &dyn LlmProvider,
        actual_rounds: usize,
        round_history: Vec<Vec<String>>,
        mut warnings: Vec<String>,
    ) -> Result<ConsensusResult> {
        let positions_text = final_positions
            .iter()
            .enumerate()
            .map(|(i, pos)| format!("--- Position {} ---\n{}", i, pos))
            .collect::<Vec<_>>()
            .join("\n\n");

        let judge_prompt = DEBATE_JUDGE_PROMPT
            .replace("{count}", &final_positions.len().to_string())
            .replace("{candidates}", &positions_text);

        let judge_response = judge.complete(&judge_prompt, Some(DEFAULT_JUDGE_SYSTEM)).await?;

        let mut metadata = HashMap::new();
        metadata.insert("rounds_completed".to_string(), serde_json::json!(actual_rounds));
        metadata.insert("round_history".to_string(), serde_json::json!(round_history));
        metadata.insert("blind".to_string(), serde_json::json!(true));

        // Map dissenting final positions back to the originating candidates.
        let dissent_candidates = |indices: &[usize]| -> Vec<Candidate> {
            indices.iter().filter_map(|&i| candidates.get(i).cloned()).collect()
        };

        let (content, agreement_score, dissents, reasoning) =
            match parse_judge_response(&judge_response) {
                Ok(parsed) => {
                    let (valid, dropped): (Vec<usize>, Vec<usize>) =
                        parsed.dissent_indices.iter().copied().partition(|&i| i < candidates.len());
                    if !dropped.is_empty() {
                        warnings.push(format!(
                            "judge returned out-of-range dissent indices {dropped:?}; dropped"
                        ));
                    }
                    metadata.insert("judge_parse".to_string(), serde_json::json!("structured"));
                    (
                        parsed.synthesis,
                        parsed.agreement_score,
                        dissent_candidates(&valid),
                        parsed.reasoning,
                    )
                }
                Err(parse_err) => {
                    // Deterministic fallback: the most central final position
                    // (highest mean similarity to the others), with honest
                    // similarity-based scoring — never a fabricated 1.0.
                    let consensus_idx = most_central_index(final_positions);
                    let scores: Vec<f64> = final_positions
                        .iter()
                        .map(|p| text_similarity(&final_positions[consensus_idx], p))
                        .collect();
                    let avg = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
                    let dissent_indices: Vec<usize> = scores
                        .iter()
                        .enumerate()
                        .filter(|&(_, &s)| s < 0.3)
                        .map(|(i, _)| i)
                        .collect();
                    metadata.insert("judge_parse".to_string(), serde_json::json!("fallback"));
                    warnings.push(format!(
                        "judge response was not valid JSON ({parse_err}); \
                         consensus picked by text similarity, agreement estimated"
                    ));
                    (
                        final_positions[consensus_idx].clone(),
                        avg,
                        dissent_candidates(&dissent_indices),
                        "Judge response was not in structured format; \
                         agreement estimated from text similarity"
                            .to_string(),
                    )
                }
            };

        if !warnings.is_empty() {
            metadata.insert("warnings".to_string(), serde_json::json!(warnings));
        }

        Ok(ConsensusResult {
            content,
            strategy: self.name().to_string(),
            agreement_score,
            candidates: candidates.to_vec(),
            dissents,
            reasoning: Some(format!(
                "{reasoning} (debate completed in {} round(s) of {} maximum)",
                actual_rounds, self.config.max_rounds,
            )),
            metadata,
        })
    }
}

impl Default for MultiRoundDebate {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the blind critique prompt for one participant: positions are
/// anonymized and shuffled with a deterministic per-participant, per-round
/// permutation so no participant can identify its own position.
fn blind_critique_prompt(
    question: &str,
    positions: &[String],
    participant_idx: usize,
    round: usize,
    max_rounds: usize,
) -> String {
    let order = shuffled_indices(positions.len(), participant_idx, round);
    let positions_text = order
        .iter()
        .enumerate()
        .map(|(label, &pos_idx)| format!("--- Position {} ---\n{}", label + 1, positions[pos_idx]))
        .collect::<Vec<_>>()
        .join("\n\n");

    BLIND_CRITIQUE_PROMPT
        .replace("{question}", question)
        .replace("{positions}", &positions_text)
        .replace("{round}", &round.to_string())
        .replace("{max_rounds}", &max_rounds.to_string())
}

/// Deterministic permutation of `0..n` seeded by participant and round.
/// (A simple xorshift — no external RNG dependency, stable across runs.)
fn shuffled_indices(n: usize, participant_idx: usize, round: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state =
        ((participant_idx as u64 + 1) << 32) ^ ((round as u64 + 1).wrapping_mul(0x9E37_79B9));
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
    indices
}

/// Extract the text after a `FINAL ANSWER:` marker; fall back to the whole
/// response when the marker is absent.
fn extract_final_answer(response: &str) -> String {
    if let Some(pos) = response.rfind("FINAL ANSWER:") {
        let answer = response[pos + "FINAL ANSWER:".len()..].trim();
        if !answer.is_empty() {
            return answer.to_string();
        }
    }
    response.trim().to_string()
}

/// Index of the position with the highest mean similarity to all others
/// (deterministic tie-break: lowest index wins).
fn most_central_index(positions: &[String]) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, p) in positions.iter().enumerate() {
        let score = positions
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, q)| text_similarity(p, q))
            .sum::<f64>();
        if score > best_score {
            best = i;
            best_score = score;
        }
    }
    best
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

    /// Single-provider debate: the same LLM refines every position, but each
    /// position is still refined independently through its own blind-critique
    /// prompt — never one broadcast rewrite applied to all.
    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let llm =
            llm.ok_or_else(|| anyhow::anyhow!("MultiRoundDebate requires an LLM provider"))?;
        let participants: Vec<&dyn LlmProvider> = candidates.iter().map(|_| llm).collect();
        self.resolve_with_participants(candidates, &participants, llm).await
    }

    /// Multi-provider debate: each candidate's position is refined by its own
    /// model's provider from `participants`, falling back to the judge LLM
    /// for candidates without a registered provider.
    async fn resolve_multi(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
        participants: Option<&MultiProvider>,
    ) -> Result<ConsensusResult> {
        let judge =
            llm.ok_or_else(|| anyhow::anyhow!("MultiRoundDebate requires an LLM provider"))?;

        let mut participant_llms: Vec<&dyn LlmProvider> = Vec::with_capacity(candidates.len());
        for c in candidates {
            let own = c.model.as_deref().and_then(|m| participants.and_then(|p| p.get(m)));
            participant_llms.push(own.unwrap_or(judge));
        }

        self.resolve_with_participants(candidates, &participant_llms, judge).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    struct FailingProvider;

    #[async_trait]
    impl LlmProvider for FailingProvider {
        async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
            anyhow::bail!("simulated participant failure")
        }
    }

    fn judge_json(score: f64) -> String {
        serde_json::json!({
            "synthesis": "The adjudicated consensus answer.",
            "reasoning": "Positions broadly agreed.",
            "agreement_score": score,
            "dissent_indices": []
        })
        .to_string()
    }

    #[tokio::test]
    async fn debate_independent_positions_not_broadcast() {
        // Two participants get different refinements; the final positions must
        // NOT collapse into one broadcast rewrite.
        let provider = MockProvider::new(vec![
            "Critique...\nFINAL ANSWER: Refined position Alpha".to_string(),
            "Critique...\nFINAL ANSWER: Refined position Beta".to_string(),
            judge_json(0.6),
        ]);
        let candidates = vec![
            Candidate::new("Answer A from model 1").with_model("model-1"),
            Candidate::new("Answer B from model 2").with_model("model-2"),
        ];

        let strategy = MultiRoundDebate::new().with_rounds(1);
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.strategy, "multi_round_debate");
        assert_eq!(result.content, "The adjudicated consensus answer.");
        // Honest scoring: the judge's reported score is kept, not inflated.
        assert_eq!(result.agreement_score, 0.6);
        let history = result.metadata["round_history"].as_array().unwrap();
        let last = history.last().unwrap().as_array().unwrap();
        let positions: Vec<&str> = last.iter().filter_map(|p| p.as_str()).collect();
        assert_eq!(positions, vec!["Refined position Alpha", "Refined position Beta"]);
        assert_eq!(result.metadata["blind"], serde_json::json!(true));
        assert_eq!(result.metadata["judge_parse"], serde_json::json!("structured"));
    }

    #[tokio::test]
    async fn debate_multi_provider_uses_each_models_own_provider() {
        // resolve_multi must route each candidate's refinement to its own provider.
        let multi = MultiProvider::new()
            .add("model-1", MockProvider::fixed("FINAL ANSWER: From provider one"))
            .add("model-2", MockProvider::fixed("FINAL ANSWER: From provider two"));
        let judge = MockProvider::fixed(judge_json(0.7));

        let candidates = vec![
            Candidate::new("Answer A").with_model("model-1"),
            Candidate::new("Answer B").with_model("model-2"),
        ];

        let strategy = MultiRoundDebate::new().with_rounds(1);
        let result = strategy.resolve_multi(&candidates, Some(&judge), Some(&multi)).await.unwrap();

        let history = result.metadata["round_history"].as_array().unwrap();
        let last = history.last().unwrap().as_array().unwrap();
        let positions: Vec<&str> = last.iter().filter_map(|p| p.as_str()).collect();
        assert_eq!(positions, vec!["From provider one", "From provider two"]);
    }

    #[tokio::test]
    async fn debate_retains_position_when_one_participant_fails() {
        let failing = FailingProvider;
        let healthy = MockProvider::fixed("FINAL ANSWER: Healthy revision");
        let judge = MockProvider::fixed(judge_json(0.5));
        let candidates = vec![Candidate::new("Original A"), Candidate::new("Original B")];
        let participants: Vec<&dyn LlmProvider> = vec![&failing, &healthy];

        let result = MultiRoundDebate::new()
            .with_rounds(1)
            .resolve_with_participants(&candidates, &participants, &judge)
            .await
            .unwrap();

        let last = result.metadata["round_history"].as_array().unwrap().last().unwrap();
        assert_eq!(last[0], serde_json::json!("Original A"));
        assert_eq!(last[1], serde_json::json!("Healthy revision"));
        let warnings = result.metadata["warnings"].as_array().unwrap();
        assert!(warnings.iter().any(|warning| warning.as_str().unwrap().contains("participant 0")));
    }

    #[tokio::test]
    async fn all_failed_participants_never_fabricate_early_convergence() {
        let failing_a = FailingProvider;
        let failing_b = FailingProvider;
        let judge = MockProvider::fixed(judge_json(0.2));
        let candidates = vec![Candidate::new("Original A"), Candidate::new("Original B")];
        let participants: Vec<&dyn LlmProvider> = vec![&failing_a, &failing_b];

        let result = MultiRoundDebate::new()
            .with_rounds(3)
            .resolve_with_participants(&candidates, &participants, &judge)
            .await
            .unwrap();

        assert_eq!(result.metadata["rounds_completed"], serde_json::json!(3));
        assert_eq!(result.metadata["round_history"].as_array().unwrap().len(), 4);
    }

    #[tokio::test]
    async fn debate_judge_fallback_is_honest() {
        let provider = MockProvider::new(vec![
            "FINAL ANSWER: Refined".to_string(),
            "FINAL ANSWER: Refined".to_string(),
            "not json at all".to_string(),
        ]);
        let candidates = vec![Candidate::new("A"), Candidate::new("A")];

        let strategy = MultiRoundDebate::new().with_rounds(1);
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.metadata["judge_parse"], serde_json::json!("fallback"));
        assert!(result.metadata.contains_key("warnings"));
        // Identical positions → similarity-based agreement of 1.0 is legitimate
        // here, but it comes from measurement, not from fabrication.
        assert_eq!(result.agreement_score, 1.0);
        assert_eq!(result.content, "Refined");
    }

    #[tokio::test]
    async fn debate_converges_early_when_positions_stabilize() {
        // Every response is identical, so round 1 already converges; the judge
        // (response 3) adjudicates. max_rounds=5 must not burn extra calls.
        let provider = MockProvider::new(vec![
            "Same answer".to_string(),
            "Same answer".to_string(),
            judge_json(0.9),
            "unused".to_string(),
        ]);
        let candidates = vec![Candidate::new("Same answer"), Candidate::new("Same answer")];

        let strategy = MultiRoundDebate::new().with_rounds(5);
        let result = strategy.resolve(&candidates, Some(&provider)).await.unwrap();

        assert_eq!(result.metadata["rounds_completed"], serde_json::json!(1));
        assert_eq!(result.agreement_score, 0.9);
    }

    #[tokio::test]
    async fn debate_requires_llm() {
        let candidates = vec![Candidate::new("test")];
        let strategy = MultiRoundDebate::new();
        let result = strategy.resolve(&candidates, None).await;
        assert!(result.is_err());
    }

    #[test]
    fn shuffle_is_deterministic_and_a_permutation() {
        let a = shuffled_indices(4, 0, 1);
        let b = shuffled_indices(4, 0, 1);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn shuffle_differs_per_participant() {
        // With enough positions, different participants should (deterministically)
        // see different orders, so positions cannot be tracked across the panel.
        let a = shuffled_indices(6, 0, 1);
        let b = shuffled_indices(6, 1, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn extract_final_answer_marker() {
        assert_eq!(extract_final_answer("critique\nFINAL ANSWER: real answer"), "real answer");
        assert_eq!(extract_final_answer("no marker here"), "no marker here");
        assert_eq!(extract_final_answer("FINAL ANSWER:\n"), "FINAL ANSWER:");
    }
}
