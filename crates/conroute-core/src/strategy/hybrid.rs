use anyhow::Result;
use async_trait::async_trait;

use crate::strategy::debate::{DebateConfig, MultiRoundDebate};
use crate::strategy::vote::MajorityVote;
use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// Hybrid strategy: N rounds of debate followed by voting.
/// Per ACL 2025 findings, this combines the strengths of debate (knowledge)
/// and voting (reasoning).
pub struct DebateThenVote {
    pub debate_rounds: usize,
    pub vote_threshold: f64,
    pub convergence_threshold: f64,
}

impl Default for DebateThenVote {
    fn default() -> Self {
        Self {
            debate_rounds: 2,
            vote_threshold: 0.8,
            convergence_threshold: 0.9,
        }
    }
}

impl DebateThenVote {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_debate_rounds(mut self, rounds: usize) -> Self {
        self.debate_rounds = rounds;
        self
    }

    pub fn with_vote_threshold(mut self, threshold: f64) -> Self {
        self.vote_threshold = threshold;
        self
    }
}

#[async_trait]
impl ConsensusStrategy for DebateThenVote {
    fn name(&self) -> &str {
        "debate_then_vote"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let llm = llm.ok_or_else(|| {
            anyhow::anyhow!("DebateThenVote requires an LLM provider")
        })?;

        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }

        // Phase 1: Debate
        let debate = MultiRoundDebate::with_config(DebateConfig {
            max_rounds: self.debate_rounds,
            convergence_threshold: self.convergence_threshold,
            ..Default::default()
        });

        let debate_result = debate.resolve(candidates, Some(llm)).await?;

        // Create new candidates from the debate output
        // The debate produces refined positions; we treat the debate output
        // plus original candidates as the voting pool
        let mut vote_candidates = vec![Candidate::new(&debate_result.content)
            .with_metadata("source", serde_json::json!("debate_synthesis"))];

        // Include original candidates that are still distinct
        for candidate in candidates {
            vote_candidates.push(candidate.clone());
        }

        // Phase 2: Vote
        let vote = MajorityVote::new().with_threshold(self.vote_threshold);
        let mut vote_result = vote.resolve(&vote_candidates, None).await?;

        // Override strategy name and add combined metadata
        vote_result.strategy = self.name().to_string();
        vote_result.metadata.insert(
            "debate_rounds".to_string(),
            serde_json::json!(self.debate_rounds),
        );
        vote_result.metadata.insert(
            "debate_agreement".to_string(),
            serde_json::json!(debate_result.agreement_score),
        );
        vote_result.reasoning = Some(format!(
            "Debate phase ({} rounds, agreement: {:.2}) followed by majority vote (agreement: {:.2})",
            self.debate_rounds, debate_result.agreement_score, vote_result.agreement_score,
        ));

        // Restore original candidates (not the inflated vote pool)
        vote_result.candidates = candidates.to_vec();

        Ok(vote_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[tokio::test]
    async fn debate_then_vote_basic() {
        let provider = MockProvider::fixed("The consensus after debate.");
        let candidates = vec![
            Candidate::new("Position A"),
            Candidate::new("Position B"),
            Candidate::new("Position C"),
        ];

        let strategy = DebateThenVote::new().with_debate_rounds(1);
        let result = strategy
            .resolve(&candidates, Some(&provider))
            .await
            .unwrap();

        assert_eq!(result.strategy, "debate_then_vote");
        assert!(!result.content.is_empty());
    }
}
