use conroute_core::{consensus, Candidate, OutputFormat};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create candidates (in real usage, these come from different LLMs)
    let candidates = vec![
        Candidate::new("The capital of France is Paris.")
            .with_model("gpt-4o")
            .with_confidence(0.95),
        Candidate::new("Paris is the capital city of France.")
            .with_model("claude-sonnet-4")
            .with_confidence(0.92),
        Candidate::new("The capital of France is Lyon.")
            .with_model("small-model")
            .with_confidence(0.3),
    ];

    // Run majority vote (no LLM needed)
    println!("=== Majority Vote ===");
    let result = consensus(&candidates, "majority_vote", None).await?;
    println!("{}", OutputFormat::SupremeCourt.render(&result));

    // Run weighted vote (uses confidence scores)
    println!("\n=== Weighted Vote ===");
    let result = consensus(&candidates, "weighted_vote", None).await?;
    println!("Winner: {}", result.content);
    println!("Agreement: {:.0}%", result.agreement_score * 100.0);
    println!("Dissents: {}", result.dissents.len());

    Ok(())
}
