use clap::Args;
use colored::Colorize;
use caucus_core::{Candidate, MultiRoundDebate, ConsensusStrategy, OutputFormat};
use caucus_core::strategy::debate::DebateConfig;

use super::{build_provider, build_single_provider, default_models};

#[derive(Args)]
pub struct DebateArgs {
    /// The question or topic for debate
    pub prompt: String,

    /// Comma-separated list of models to participate (defaults to all configured providers)
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Number of debate rounds
    #[arg(short, long, default_value = "3")]
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

    // Generate initial positions
    let provider = build_provider(&models)?;
    let mut candidates = Vec::new();

    for model in &models {
        let llm = provider
            .get(model)
            .ok_or_else(|| anyhow::anyhow!("No provider for model: {model}"))?;

        eprintln!("  {} Getting initial position from {}...", "·".dimmed(), model.yellow());
        let response = llm.complete(&args.prompt, None).await?;
        eprintln!(
            "  {} {} responded ({} chars)",
            "✓".green(),
            model,
            response.len(),
        );
        candidates.push(
            Candidate::new(response)
                .with_model(model.clone())
                .with_metadata("question", serde_json::json!(&args.prompt)),
        );
    }

    eprintln!();

    // Run debate
    let judge_model = models.first().unwrap();
    let judge_llm = build_single_provider(judge_model)?;

    let strategy = MultiRoundDebate::with_config(DebateConfig {
        max_rounds: args.rounds,
        ..Default::default()
    });

    let result = strategy.resolve(&candidates, Some(judge_llm.as_ref())).await?;

    eprintln!(
        "\n{} Debate concluded (agreement: {:.0}%)\n",
        "✓".green(),
        result.agreement_score * 100.0,
    );

    println!("{}", format.render(&result));

    Ok(())
}
