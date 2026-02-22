use crate::types::ConsensusResult;

/// Render a consensus result in "Supreme Court" format:
/// majority opinion, concurrences, and dissents.
pub fn render(result: &ConsensusResult) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "═══════════════════════════════════════════\n\
         CONSENSUS OPINION — {} Strategy\n\
         Agreement: {:.0}%\n\
         ═══════════════════════════════════════════\n\n",
        result.strategy,
        result.agreement_score * 100.0,
    ));

    // Majority Opinion
    output.push_str("MAJORITY OPINION\n");
    output.push_str("───────────────────────────────────────────\n");
    output.push_str(&result.content);
    output.push_str("\n\n");

    // Reasoning
    if let Some(reasoning) = &result.reasoning {
        output.push_str("REASONING\n");
        output.push_str("───────────────────────────────────────────\n");
        output.push_str(reasoning);
        output.push_str("\n\n");
    }

    // Concurrences (candidates that agreed with the majority)
    let concurrences: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            !result
                .dissents
                .iter()
                .any(|d| d.content == c.content)
                && c.content != result.content
        })
        .collect();

    if !concurrences.is_empty() {
        output.push_str(&format!("CONCURRENCES ({})\n", concurrences.len()));
        output.push_str("───────────────────────────────────────────\n");
        for (i, c) in concurrences.iter().enumerate() {
            let model = c.model.as_deref().unwrap_or("Anonymous");
            output.push_str(&format!("{}. {} wrote:\n", i + 1, model));
            output.push_str(&format!("   {}\n\n", c.content));
        }
    }

    // Dissents
    if !result.dissents.is_empty() {
        output.push_str(&format!("DISSENTS ({})\n", result.dissents.len()));
        output.push_str("───────────────────────────────────────────\n");
        for (i, d) in result.dissents.iter().enumerate() {
            let model = d.model.as_deref().unwrap_or("Anonymous");
            output.push_str(&format!("{}. {} wrote:\n", i + 1, model));
            output.push_str(&format!("   {}\n\n", d.content));
        }
    }

    // Vote summary
    output.push_str("VOTE SUMMARY\n");
    output.push_str("───────────────────────────────────────────\n");
    output.push_str(&format!(
        "Total candidates: {}\n\
         In agreement:     {}\n\
         Dissenting:       {}\n\
         Agreement score:  {:.1}%\n",
        result.candidates.len(),
        result.candidates.len() - result.dissents.len(),
        result.dissents.len(),
        result.agreement_score * 100.0,
    ));

    output
}
