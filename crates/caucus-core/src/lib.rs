//! caucus-core: Multi-LLM consensus engine.
//!
//! Provides composable strategies for aggregating and synthesizing outputs
//! from multiple LLMs into consensus results.
//!
//! # Quick Start
//!
//! ```rust
//! use caucus_core::{Candidate, consensus};
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

pub mod adapters;
pub mod config;
pub mod error;
pub mod fanout;
pub mod format;
pub mod pipeline;
pub mod process;
pub mod provider;
pub mod strategy;
pub mod types;

// Re-export primary types at the crate root for convenience.
pub use adapters::{
    AcpProvider, AdapterDescriptor, AdapterOverrides, CommandInvocation, CommandProvider,
    CommandSpec, Discovered, Effort, KIMI_EFFORT_ENV, MemberSpec, PromptDelivery, Readiness,
    Stability, Utility, build_invocation, build_invocation_with, descriptor, descriptors,
    provider_for, provider_for_with, utility_ids, validate_effort,
};
pub use config::{Config, ConfigError, Council, Profile, builtin_profiles};
pub use error::{ErrorKind, ProviderError, from_reqwest};
pub use fanout::{FailureRecord, FanoutBatch, bounded_fanout};
pub use pipeline::{Pipeline, VoteMethod, consensus, strategy_from_name};
pub use process::{
    ProcessLimits, ProcessOutput, ProcessSpec, SAFE_ENV_ALLOWLIST, SAFE_ENV_PREFIXES, find_on_path,
    run_argv,
};
pub use provider::{
    DEFAULT_REQUEST_TIMEOUT, FanoutConfig, FanoutFailure, FanoutReport, FanoutSuccess,
    HttpProvider, MockProvider, MultiProvider,
};
pub use types::{
    Candidate, CompleteOutcome, ConsensusResult, ConsensusStrategy, LlmProvider, ProviderOptions,
    ResponseMeta, Transport,
};

pub use format::OutputFormat;
pub use strategy::{
    DebateThenVote, JudgeSynthesis, MajorityVote, MultiRoundDebate, SemanticClustering,
    WeightedVote,
};
