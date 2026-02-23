pub mod debate;
pub mod hybrid;
pub mod judge;
pub mod semantic;
pub mod vote;

pub use debate::MultiRoundDebate;
pub use hybrid::DebateThenVote;
pub use judge::JudgeSynthesis;
pub use semantic::SemanticClustering;
pub use vote::{MajorityVote, WeightedVote};
