//! conroute-core: Multi-LLM consensus engine.
//!
//! Provides composable strategies for aggregating and synthesizing outputs
//! from multiple LLMs into consensus results.
//!
//! # Quick Start
//!
//! ```rust
//! use conroute_core::{Candidate, consensus};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let candidates = vec![
//!     Candidate::new("The answer is 42"),
//!     Candidate::new("The answer is 42"),
//!     Candidate::new("The answer is 7"),
//! ];
//!
//! let result = consensus(&candidates, "majority_vote", None).await?;
//! println!("Consensus: {}", result.content);
//! println!("Agreement: {:.0}%", result.agreement_score * 100.0);
//! # Ok(())
//! # }
//! ```

pub mod format;
pub mod pipeline;
pub mod provider;
pub mod strategy;
pub mod types;

// Re-export primary types at the crate root for convenience.
pub use pipeline::{consensus, strategy_from_name, Pipeline, VoteMethod};
pub use provider::{HttpProvider, MockProvider, MultiProvider};
pub use types::{Candidate, ConsensusResult, ConsensusStrategy, LlmProvider};

pub use format::OutputFormat;
pub use strategy::{
    DebateThenVote, JudgeSynthesis, MajorityVote, MultiRoundDebate, SemanticClustering,
    WeightedVote,
};
