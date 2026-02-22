use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

/// Cluster responses by embedding similarity, then select or synthesize from
/// the dominant cluster. Requires an LLM provider that supports embeddings.
pub struct SemanticClustering {
    /// Minimum cosine similarity for two candidates to be in the same cluster.
    pub similarity_threshold: f64,
    /// If true, synthesize a combined response from the winning cluster.
    /// If false, select the centroid-nearest candidate.
    pub synthesize: bool,
}

impl Default for SemanticClustering {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.75,
            synthesize: false,
        }
    }
}

impl SemanticClustering {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    pub fn with_synthesis(mut self, synthesize: bool) -> Self {
        self.synthesize = synthesize;
        self
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn centroid(embeddings: &[&Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dim = embeddings[0].len();
    let mut center = vec![0.0; dim];
    for emb in embeddings {
        for (i, val) in emb.iter().enumerate() {
            center[i] += val;
        }
    }
    let n = embeddings.len() as f64;
    center.iter_mut().for_each(|v| *v /= n);
    center
}

#[async_trait]
impl ConsensusStrategy for SemanticClustering {
    fn name(&self) -> &str {
        "semantic_clustering"
    }

    async fn resolve(
        &self,
        candidates: &[Candidate],
        llm: Option<&dyn LlmProvider>,
    ) -> Result<ConsensusResult> {
        let llm = llm.ok_or_else(|| {
            anyhow::anyhow!("SemanticClustering requires an LLM provider with embedding support")
        })?;

        if candidates.is_empty() {
            anyhow::bail!("No candidates provided");
        }

        // Get embeddings for all candidates
        let texts: Vec<String> = candidates.iter().map(|c| c.content.clone()).collect();
        let embeddings = llm.embed(&texts).await?;

        if embeddings.len() != candidates.len() {
            anyhow::bail!(
                "Got {} embeddings for {} candidates",
                embeddings.len(),
                candidates.len()
            );
        }

        // Cluster by greedy similarity
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for (i, emb) in embeddings.iter().enumerate() {
            let mut placed = false;
            for cluster in &mut clusters {
                let cluster_embs: Vec<&Vec<f64>> =
                    cluster.iter().map(|&idx| &embeddings[idx]).collect();
                let center = centroid(&cluster_embs);
                if cosine_similarity(emb, &center) >= self.similarity_threshold {
                    cluster.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                clusters.push(vec![i]);
            }
        }

        // Find the largest cluster
        clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let winning_cluster = &clusters[0];
        let agreement_score = winning_cluster.len() as f64 / candidates.len() as f64;

        // Find centroid-nearest candidate in the winning cluster
        let cluster_embs: Vec<&Vec<f64>> = winning_cluster
            .iter()
            .map(|&idx| &embeddings[idx])
            .collect();
        let center = centroid(&cluster_embs);

        let best_idx = *winning_cluster
            .iter()
            .max_by(|&&a, &&b| {
                cosine_similarity(&embeddings[a], &center)
                    .partial_cmp(&cosine_similarity(&embeddings[b], &center))
                    .unwrap()
            })
            .unwrap();

        let content = if self.synthesize && winning_cluster.len() > 1 {
            // Ask the LLM to synthesize from the winning cluster
            let cluster_texts: Vec<String> = winning_cluster
                .iter()
                .map(|&i| {
                    format!(
                        "--- Response {} ---\n{}",
                        i + 1,
                        candidates[i].content.clone()
                    )
                })
                .collect();

            let prompt = format!(
                "The following responses were identified as semantically similar. \
                 Synthesize them into a single, comprehensive response:\n\n{}",
                cluster_texts.join("\n\n")
            );

            llm.complete(&prompt, None).await?
        } else {
            candidates[best_idx].content.clone()
        };

        let winning_set: std::collections::HashSet<usize> =
            winning_cluster.iter().copied().collect();
        let dissents: Vec<Candidate> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| !winning_set.contains(i))
            .map(|(_, c)| c.clone())
            .collect();

        Ok(ConsensusResult {
            content,
            strategy: self.name().to_string(),
            agreement_score,
            candidates: candidates.to_vec(),
            dissents,
            reasoning: Some(format!(
                "Found {} clusters; dominant cluster has {} of {} candidates",
                clusters.len(),
                winning_cluster.len(),
                candidates.len(),
            )),
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
