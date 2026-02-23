use crate::types::ConsensusResult;

/// Maximum number of lines to show per concurrence/dissent preview.
const PREVIEW_LINES: usize = 5;

/// Truncate text to a preview: first N non-empty lines, with a note if truncated.
fn preview(text: &str, max_lines: usize) -> (String, bool) {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() <= max_lines {
        (lines.join("\n"), false)
    } else {
        (lines[..max_lines].join("\n"), true)
    }
}

/// Render a consensus result in "Supreme Court" format:
/// majority opinion, concurrences (truncated), and dissents (truncated).
pub fn render(result: &ConsensusResult) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "\n\
         ═══════════════════════════════════════════════════\n  \
         CONSENSUS OPINION — {} Strategy\n  \
         Agreement: {:.0}%\n\
         ═══════════════════════════════════════════════════\n\n",
        result.strategy,
        result.agreement_score * 100.0,
    ));

    // Majority Opinion
    output.push_str("  MAJORITY OPINION\n");
    output.push_str("  ─────────────────────────────────────────────────\n\n");
    output.push_str(&result.content);
    output.push_str("\n\n");

    // Reasoning
    if let Some(reasoning) = &result.reasoning {
        output.push_str("═══════════════════════════════════════════════════\n");
        output.push_str("  REASONING\n");
        output.push_str("═══════════════════════════════════════════════════\n\n");
        output.push_str(reasoning);
        output.push_str("\n\n");
    }

    // Concurrences (candidates that agreed with the majority)
    let concurrences: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            !result.dissents.iter().any(|d| d.content == c.content) && c.content != result.content
        })
        .collect();

    if !concurrences.is_empty() {
        output.push_str("═══════════════════════════════════════════════════\n");
        output.push_str(&format!("  CONCURRENCES ({})\n", concurrences.len()));
        output.push_str("═══════════════════════════════════════════════════\n\n");
        for (i, c) in concurrences.iter().enumerate() {
            let model = c.model.as_deref().unwrap_or("Anonymous");
            let (text, truncated) = preview(&c.content, PREVIEW_LINES);
            output.push_str(&format!("  {}. {} wrote:\n\n", i + 1, model));
            for line in text.lines() {
                output.push_str(&format!("     {}\n", line));
            }
            if truncated {
                output.push_str("\n     ... [truncated — use -f detailed for full text]\n");
            }
            output.push('\n');
        }
    }

    // Dissents
    if !result.dissents.is_empty() {
        output.push_str("═══════════════════════════════════════════════════\n");
        output.push_str(&format!("  DISSENTS ({})\n", result.dissents.len()));
        output.push_str("═══════════════════════════════════════════════════\n\n");
        for (i, d) in result.dissents.iter().enumerate() {
            let model = d.model.as_deref().unwrap_or("Anonymous");
            let (text, truncated) = preview(&d.content, PREVIEW_LINES);
            output.push_str(&format!("  {}. {} wrote:\n\n", i + 1, model));
            for line in text.lines() {
                output.push_str(&format!("     {}\n", line));
            }
            if truncated {
                output.push_str("\n     ... [truncated — use -f detailed for full text]\n");
            }
            output.push('\n');
        }
    }

    // Vote summary
    output.push_str("═══════════════════════════════════════════════════\n");
    output.push_str("  VOTE SUMMARY\n");
    output.push_str("═══════════════════════════════════════════════════\n\n");
    output.push_str(&format!(
        "  Total candidates: {}\n\
         \x20 In agreement:     {}\n\
         \x20 Dissenting:       {}\n\
         \x20 Agreement score:  {:.1}%\n",
        result.candidates.len(),
        result.candidates.len() - result.dissents.len(),
        result.dissents.len(),
        result.agreement_score * 100.0,
    ));

    output
}
