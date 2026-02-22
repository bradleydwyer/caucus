use crate::types::ConsensusResult;

/// Render a consensus result as pretty-printed JSON.
pub fn render(result: &ConsensusResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("JSON serialization error: {e}"))
}
