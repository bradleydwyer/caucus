use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// Count-based majority voting with fuzzy string matching.
///
/// Groups candidates by normalized content similarity, then selects the
/// largest group as the consensus. No LLM required.
pub struct MajorityVote {
    /// Similarity threshold (0.0–1.0) for grouping candidates.
    /// Candidates with normalized similarity >= threshold are grouped together.
    pub similarity_threshold: f64,
}

impl Default for MajorityVote {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.8,
        }
    }
}

impl MajorityVote {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}

/// Compute normalized similarity between two strings using bigram overlap.
fn bigram_similarity(a: &str, b: &str) -> f64 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();

    if a == b {
        return 1.0;
    }
    if a.len() < 2 || b.len() < 2 {
        return if a == b { 1.0 } else { 0.0 };
    }

    let bigrams_a: Vec<(char, char)> = a.chars().zip(a.chars().skip(1)).collect();
    let bigrams_b: Vec<(char, char)> = b.chars().zip(b.chars().skip(1)).collect();

    if bigrams_a.is_empty() || bigrams_b.is_empty() {
        return 0.0;
    }

    let matches = bigrams_a
        .iter()
        .filter(|bg| bigrams_b.contains(bg))
        .count();

    (2.0 * matches as f64) / (bigrams_a.len() + bigrams_b.len()) as f64
}

#[async_trait]
impl ConsensusStrategy for MajorityVote {
    fn name(&self) -> &str {
        "majority_vote"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        _llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }
        if candidates.len() == 1 {
            return Ok(ConsensusResult {
                content: candidates[0].content.clone(),
                strategy: self.name().to_string(),
                agreement_score: 1.0,
                candidates: candidates.to_vec(),
                dissents: vec![],
                reasoning: Some("Only one candidate provided".to_string()),
                metadata: HashMap::new(),
            });
        }

        // Group candidates by similarity
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            let mut placed = false;
            for group in &mut groups {
                let representative = &candidates[group[0]];
                if bigram_similarity(&candidate.content, &representative.content)
                    >= self.similarity_threshold
                {
                    group.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![i]);
            }
        }

        // Find the largest group
        groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
        let winning_group = &groups[0];
        let total = candidates.len();
        let agreement_score = winning_group.len() as f64 / total as f64;

        // Use the first candidate in the winning group as the representative
        let winner = &candidates[winning_group[0]];

        // Collect dissents (candidates not in the winning group)
        let winning_set: std::collections::HashSet<usize> =
            winning_group.iter().copied().collect();
        let dissents: Vec<Candidate> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| !winning_set.contains(i))
            .map(|(_, c)| c.clone())
            .collect();

        Ok(ConsensusResult {
            content: winner.content.clone(),
            strategy: self.name().to_string(),
            agreement_score,
            candidates: candidates.to_vec(),
            dissents,
            reasoning: Some(format!(
                "{} of {} candidates agreed (threshold: {:.0}%)",
                winning_group.len(),
                total,
                self.similarity_threshold * 100.0,
            )),
            metadata: HashMap::new(),
        })
    }
}

/// Weighted voting where candidates contribute based on confidence or model reputation.
pub struct WeightedVote {
    /// Similarity threshold for grouping.
    pub similarity_threshold: f64,
    /// Default weight for candidates without explicit confidence.
    pub default_weight: f64,
    /// Optional per-model weight overrides.
    pub model_weights: HashMap<String, f64>,
}

impl Default for WeightedVote {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.8,
            default_weight: 1.0,
            model_weights: HashMap::new(),
        }
    }
}

impl WeightedVote {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    pub fn with_model_weight(mut self, model: impl Into<String>, weight: f64) -> Self {
        self.model_weights.insert(model.into(), weight);
        self
    }

    fn weight_for(&self, candidate: &Candidate) -> f64 {
        // Priority: confidence > model weight > default
        if let Some(conf) = candidate.confidence {
            return conf;
        }
        if let Some(model) = &candidate.model
            && let Some(w) = self.model_weights.get(model)
        {
            return *w;
        }
        self.default_weight
    }
}

#[async_trait]
impl ConsensusStrategy for WeightedVote {
    fn name(&self) -> &str {
        "weighted_vote"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        _llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }
        if candidates.len() == 1 {
            return Ok(ConsensusResult {
                content: candidates[0].content.clone(),
                strategy: self.name().to_string(),
                agreement_score: 1.0,
                candidates: candidates.to_vec(),
                dissents: vec![],
                reasoning: Some("Only one candidate provided".to_string()),
                metadata: HashMap::new(),
            });
        }

        // Group candidates by similarity, track weighted scores
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            let mut placed = false;
            for group in &mut groups {
                let representative = &candidates[group[0]];
                if bigram_similarity(&candidate.content, &representative.content)
                    >= self.similarity_threshold
                {
                    group.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![i]);
            }
        }

        // Score each group by sum of weights
        let mut group_scores: Vec<(f64, &Vec<usize>)> = groups
            .iter()
            .map(|group| {
                let score: f64 = group.iter().map(|&i| self.weight_for(&candidates[i])).sum();
                (score, group)
            })
            .collect();
        group_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let (winning_score, winning_group) = &group_scores[0];
        let total_weight: f64 = candidates.iter().map(|c| self.weight_for(c)).sum();
        let agreement_score = winning_score / total_weight;

        // Best candidate in winning group (highest individual weight)
        let best_idx = *winning_group
            .iter()
            .max_by(|&&a, &&b| {
                self.weight_for(&candidates[a])
                    .partial_cmp(&self.weight_for(&candidates[b]))
                    .unwrap()
            })
            .unwrap();

        let winning_set: std::collections::HashSet<usize> =
            winning_group.iter().copied().collect();
        let dissents: Vec<Candidate> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| !winning_set.contains(i))
            .map(|(_, c)| c.clone())
            .collect();

        Ok(ConsensusResult {
            content: candidates[best_idx].content.clone(),
            strategy: self.name().to_string(),
            agreement_score,
            candidates: candidates.to_vec(),
            dissents,
            reasoning: Some(format!(
                "Weighted score {:.2} of {:.2} total ({}  of {} candidates)",
                winning_score,
                total_weight,
                winning_group.len(),
                candidates.len(),
            )),
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn majority_vote_unanimous() {
        let candidates = vec![
            Candidate::new("The answer is 42"),
            Candidate::new("The answer is 42"),
            Candidate::new("The answer is 42"),
        ];
        let strategy = MajorityVote::new();
        let result = strategy.resolve(&candidates, None).await.unwrap();
        assert_eq!(result.agreement_score, 1.0);
        assert!(result.dissents.is_empty());
    }

    #[tokio::test]
    async fn majority_vote_split() {
        let candidates = vec![
            Candidate::new("The answer is 42"),
            Candidate::new("The answer is 42"),
            Candidate::new("The answer is definitely 7"),
        ];
        let strategy = MajorityVote::new();
        let result = strategy.resolve(&candidates, None).await.unwrap();
        assert!(result.agreement_score > 0.5);
        assert_eq!(result.dissents.len(), 1);
    }

    #[tokio::test]
    async fn weighted_vote_prefers_confidence() {
        let candidates = vec![
            Candidate::new("Answer A").with_confidence(0.9),
            Candidate::new("Answer B completely different").with_confidence(0.1),
        ];
        let strategy = WeightedVote::new().with_threshold(0.3);
        let result = strategy.resolve(&candidates, None).await.unwrap();
        assert_eq!(result.content, "Answer A");
    }

    #[tokio::test]
    async fn majority_vote_empty_candidates() {
        let strategy = MajorityVote::new();
        let result = strategy.resolve(&[], None).await;
        assert!(result.is_err());
    }
}
