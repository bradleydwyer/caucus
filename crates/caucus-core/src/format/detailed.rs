use crate::types::ConsensusResult;

/// Render a full detailed view: all candidates, metadata, and process info.
/// Useful for debugging and research.
pub fn render(result: &ConsensusResult) -> String {
    let mut output = String::new();

    output.push_str(&format!("Strategy: {}\n", result.strategy));
    output.push_str(&format!("Agreement: {:.1}%\n\n", result.agreement_score * 100.0));

    output.push_str("=== CONSENSUS ===\n");
    output.push_str(&result.content);
    output.push_str("\n\n");

    if let Some(reasoning) = &result.reasoning {
        output.push_str("=== REASONING ===\n");
        output.push_str(reasoning);
        output.push_str("\n\n");
    }

    output.push_str("=== CANDIDATES ===\n");
    for (i, c) in result.candidates.iter().enumerate() {
        let model = c.model.as_deref().unwrap_or("unknown");
        let conf = c.confidence.map(|v| format!(" (confidence: {v:.2})")).unwrap_or_default();
        output.push_str(&format!("[{}] model={}{}\n", i + 1, model, conf));
        output.push_str(&c.content);
        output.push_str("\n\n");
    }

    if !result.dissents.is_empty() {
        output.push_str("=== DISSENTS ===\n");
        for (i, d) in result.dissents.iter().enumerate() {
            let model = d.model.as_deref().unwrap_or("unknown");
            output.push_str(&format!("[{}] model={}\n", i + 1, model));
            output.push_str(&d.content);
            output.push_str("\n\n");
        }
    }

    if !result.metadata.is_empty() {
        output.push_str("=== METADATA ===\n");
        output.push_str(
            &serde_json::to_string_pretty(&result.metadata).unwrap_or_else(|_| "{}".to_string()),
        );
        output.push('\n');
    }

    output
}
