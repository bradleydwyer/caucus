use crate::types::ConsensusResult;

/// Render a consensus result as plain text — just the consensus content.
pub fn render(result: &ConsensusResult) -> String {
    result.content.clone()
}
