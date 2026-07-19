use caucus_core::strategy::debate::DebateConfig;
use caucus_core::{Candidate, ConsensusStrategy, FanoutConfig, MultiRoundDebate, OutputFormat};
use clap::Args;
use colored::Colorize;

use super::{build_provider, default_models};

#[derive(Args)]
pub struct DebateArgs {
    /// The question or topic for debate
    pub prompt: String,

    /// Comma-separated list of models to participate (defaults to all configured providers)
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Number of debate rounds
    #[arg(short, long, default_value = "3", value_parser = super::parse_rounds)]
    pub rounds: usize,

    /// Output format
    #[arg(short, long, default_value = "detailed",
        value_parser = ["plain", "json", "supreme-court", "detailed"])]
    pub format: String,
}

pub async fn run(args: DebateArgs) -> anyhow::Result<()> {
    let format: OutputFormat = args.format.parse()?;
    let models = args.models.unwrap_or_else(default_models);

    eprintln!(
        "{} Starting debate: {} rounds with {} model(s)\n",
        "▶".green(),
        args.rounds,
        models.len(),
    );

    // Generate initial positions concurrently. A single unavailable member is
    // a recorded warning rather than aborting an otherwise viable debate.
    let provider = build_provider(&models)?;
    let report = caucus_core::provider::fanout(
        &provider,
        &args.prompt,
        None,
        FanoutConfig { quorum: 1, ..Default::default() },
    )
    .await;
    for warning in report.warnings() {
        eprintln!("  {} {}", "✗".red(), warning);
    }
    if !report.quorum_met() {
        anyhow::bail!("no debate participant returned an initial position");
    }
    let candidates: Vec<Candidate> = report
        .successes
        .iter()
        .map(|success| {
            eprintln!(
                "  {} {} responded ({} chars)",
                "✓".green(),
                success.model,
                success.content.len(),
            );
            Candidate::new(success.content.clone())
                .with_model(success.model.clone())
                .with_metadata("question", serde_json::json!(&args.prompt))
        })
        .collect();

    eprintln!();

    // Run debate
    let judge_model = candidates
        .first()
        .and_then(|candidate| candidate.model.as_deref())
        .ok_or_else(|| anyhow::anyhow!("no successful participant available as judge"))?;
    let judge_llm = provider
        .get(judge_model)
        .ok_or_else(|| anyhow::anyhow!("successful debate provider '{judge_model}' disappeared"))?;

    let strategy = MultiRoundDebate::with_config(DebateConfig {
        max_rounds: args.rounds,
        ..Default::default()
    });

    let result = strategy.resolve_multi(&candidates, Some(judge_llm), Some(&provider)).await?;

    eprintln!(
        "\n{} Debate concluded (agreement: {:.0}%)\n",
        "✓".green(),
        result.agreement_score * 100.0,
    );

    println!("{}", format.render(&result));

    Ok(())
}
