use anyhow::Result;

use crate::provider::MultiProvider;
use crate::strategy::debate::DebateConfig;
use crate::strategy::{
    DebateThenVote, JudgeSynthesis, MajorityVote, MultiRoundDebate, WeightedVote,
};
use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// A composable pipeline for multi-LLM consensus.
///
/// Each step transforms candidates or produces a consensus result.
/// Steps are executed sequentially in the order they were added.
pub struct Pipeline {
    steps: Vec<PipelineStep>,
}

enum PipelineStep {
    Generate {
        models: Vec<String>,
    },
    Debate {
        max_rounds: usize,
        convergence_threshold: f64,
    },
    Vote {
        method: VoteMethod,
    },
    Judge,
    Synthesize,
}

#[derive(Clone)]
pub enum VoteMethod {
    Majority,
    Weighted,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a generate step: query multiple models for the same prompt.
    pub fn generate(mut self, models: Vec<String>) -> Self {
        self.steps.push(PipelineStep::Generate { models });
        self
    }

    /// Add a debate step: candidates debate over multiple rounds.
    pub fn debate(mut self, max_rounds: usize) -> Self {
        self.steps.push(PipelineStep::Debate {
            max_rounds,
            convergence_threshold: 0.9,
        });
        self
    }

    /// Add a debate step with custom convergence threshold.
    pub fn debate_with_convergence(mut self, max_rounds: usize, threshold: f64) -> Self {
        self.steps.push(PipelineStep::Debate {
            max_rounds,
            convergence_threshold: threshold,
        });
        self
    }

    /// Add a vote step.
    pub fn vote(mut self, method: VoteMethod) -> Self {
        self.steps.push(PipelineStep::Vote { method });
        self
    }

    /// Add a judge synthesis step.
    pub fn judge(mut self) -> Self {
        self.steps.push(PipelineStep::Judge);
        self
    }

    /// Add a final synthesis step (same as judge but conceptually a final pass).
    pub fn synthesize(mut self) -> Self {
        self.steps.push(PipelineStep::Synthesize);
        self
    }

    /// Run the pipeline with a prompt and an LLM provider.
    pub async fn run(
        &self,
        prompt: &str,
        provider: &MultiProvider,
        judge_llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut last_result: Option<ConsensusResult> = None;

        for step in &self.steps {
            match step {
                PipelineStep::Generate { models } => {
                    candidates = generate_step(prompt, models, provider).await?;
                }
                PipelineStep::Debate {
                    max_rounds,
                    convergence_threshold,
                } => {
                    let strategy = MultiRoundDebate::with_config(DebateConfig {
                        max_rounds: *max_rounds,
                        convergence_threshold: *convergence_threshold,
                        ..Default::default()
                    });
                    let llm = judge_llm.ok_or_else(|| {
                        anyhow::anyhow!("Debate step requires a judge LLM")
                    })?;
                    let result = strategy.resolve(&candidates, Some(llm)).await?;
                    candidates = vec![Candidate::new(&result.content)
                        .with_metadata("source", serde_json::json!("debate"))];
                    // Keep original candidates too
                    for c in &result.candidates {
                        candidates.push(c.clone());
                    }
                    last_result = Some(result);
                }
                PipelineStep::Vote { method } => {
                    let result = match method {
                        VoteMethod::Majority => {
                            MajorityVote::new().resolve(&candidates, None).await?
                        }
                        VoteMethod::Weighted => {
                            WeightedVote::new().resolve(&candidates, None).await?
                        }
                    };
                    last_result = Some(result);
                }
                PipelineStep::Judge | PipelineStep::Synthesize => {
                    let llm = judge_llm.ok_or_else(|| {
                        anyhow::anyhow!("Judge/Synthesize step requires a judge LLM")
                    })?;
                    let result = JudgeSynthesis::new()
                        .resolve(&candidates, Some(llm))
                        .await?;
                    last_result = Some(result);
                }
            }
        }

        last_result.ok_or_else(|| anyhow::anyhow!("Pipeline produced no result (no steps?)"))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

async fn generate_step(
    prompt: &str,
    models: &[String],
    provider: &MultiProvider,
) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::new();

    for model in models {
        let llm = provider
            .get(model)
            .ok_or_else(|| anyhow::anyhow!("No provider registered for model: {model}"))?;

        let response = llm.complete(prompt, None).await?;
        candidates.push(
            Candidate::new(response)
                .with_model(model.clone())
                .with_metadata("question", serde_json::json!(prompt)),
        );
    }

    Ok(candidates)
}

/// Convenience function: run consensus on pre-existing candidates with a named strategy.
pub async fn consensus(
    candidates: &[Candidate],
    strategy_name: &str,
    llm: Option<&dyn LlmProvider>,
) -> Result<ConsensusResult> {
    let strategy = strategy_from_name(strategy_name)?;
    strategy.resolve(candidates, llm).await
}

/// Get a strategy instance from a name string.
pub fn strategy_from_name(name: &str) -> Result<Box<dyn ConsensusStrategy>> {
    match name {
        "majority_vote" | "majority-vote" => Ok(Box::new(MajorityVote::new())),
        "weighted_vote" | "weighted-vote" => Ok(Box::new(WeightedVote::new())),
        "judge" | "judge_synthesis" | "judge-synthesis" => Ok(Box::new(JudgeSynthesis::new())),
        "debate" | "multi_round_debate" | "multi-round-debate" => {
            Ok(Box::new(MultiRoundDebate::new()))
        }
        "debate_then_vote" | "debate-then-vote" => Ok(Box::new(DebateThenVote::new())),
        other => anyhow::bail!("Unknown strategy: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[tokio::test]
    async fn consensus_convenience_function() {
        let candidates = vec![
            Candidate::new("Answer A"),
            Candidate::new("Answer A"),
            Candidate::new("Answer B"),
        ];
        let result = consensus(&candidates, "majority_vote", None).await.unwrap();
        assert_eq!(result.strategy, "majority_vote");
        assert!(result.agreement_score > 0.5);
    }

    #[tokio::test]
    async fn pipeline_generate_and_vote() {
        let provider = MultiProvider::new()
            .add("model-a", MockProvider::fixed("The answer is 42"))
            .add("model-b", MockProvider::fixed("The answer is 42"))
            .add("model-c", MockProvider::fixed("Something else entirely"));

        let pipeline = Pipeline::new()
            .generate(vec![
                "model-a".into(),
                "model-b".into(),
                "model-c".into(),
            ])
            .vote(VoteMethod::Majority);

        let result = pipeline.run("What is the answer?", &provider, None).await.unwrap();
        assert_eq!(result.content, "The answer is 42");
    }

    #[tokio::test]
    async fn strategy_from_name_valid() {
        assert!(strategy_from_name("majority_vote").is_ok());
        assert!(strategy_from_name("majority-vote").is_ok());
        assert!(strategy_from_name("judge").is_ok());
        assert!(strategy_from_name("debate").is_ok());
        assert!(strategy_from_name("debate_then_vote").is_ok());
    }

    #[tokio::test]
    async fn strategy_from_name_invalid() {
        assert!(strategy_from_name("nonexistent").is_err());
    }
}
