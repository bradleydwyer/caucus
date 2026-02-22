use clap::Args;
use colored::Colorize;
use caucus_core::{consensus, Candidate, OutputFormat};

use super::{build_provider, build_single_provider, default_models};

#[derive(Args)]
pub struct CompareArgs {
    /// The question or prompt to send to models
    pub prompt: String,

    /// Comma-separated list of models to query (defaults to all configured providers)
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Comma-separated list of strategies to compare
    #[arg(short, long, value_delimiter = ',', default_value = "majority-vote,weighted-vote",
        value_parser = ["majority-vote", "weighted-vote", "judge", "debate", "debate-then-vote"])]
    pub strategies: Vec<String>,

    /// Output format
    #[arg(short, long, default_value = "plain",
        value_parser = ["plain", "json", "supreme-court", "detailed"])]
    pub format: String,
}

pub async fn run(args: CompareArgs) -> anyhow::Result<()> {
    let format = OutputFormat::from_str(&args.format)?;
    let models = args.models.unwrap_or_else(default_models);

    eprintln!(
        "{} Comparing {} strategies across {} model(s)...\n",
        "▶".green(),
        args.strategies.len(),
        models.len(),
    );

    // Generate candidates once
    let provider = build_provider(&models)?;
    let mut candidates = Vec::new();

    for model in &models {
        let llm = provider
            .get(model)
            .ok_or_else(|| anyhow::anyhow!("No provider for model: {model}"))?;

        eprintln!("  {} Querying {}...", "·".dimmed(), model.yellow());
        match llm.complete(&args.prompt, None).await {
            Ok(response) => {
                candidates.push(
                    Candidate::new(response)
                        .with_model(model.clone())
                        .with_metadata("question", serde_json::json!(&args.prompt)),
                );
            }
            Err(e) => {
                eprintln!("  {} {} failed: {}", "✗".red(), model, e);
            }
        }
    }

    if candidates.is_empty() {
        anyhow::bail!("No candidates generated");
    }

    // Build a judge LLM for strategies that need it
    let judge_llm: Option<Box<dyn caucus_core::LlmProvider>> = {
        let judge_model = models.first().unwrap();
        build_single_provider(judge_model).ok()
    };

    eprintln!();

    // Run each strategy
    for strategy_name in &args.strategies {
        eprintln!(
            "{} Running strategy: {}",
            "▶".green(),
            strategy_name.cyan(),
        );

        let result = consensus(&candidates, strategy_name, judge_llm.as_deref()).await;

        match result {
            Ok(result) => {
                println!(
                    "━━━ {} (agreement: {:.0}%) ━━━",
                    strategy_name.bold(),
                    result.agreement_score * 100.0,
                );
                println!("{}", format.render(&result));
                println!();
            }
            Err(e) => {
                println!("━━━ {} ━━━", strategy_name.bold().red());
                println!("Error: {e}\n");
            }
        }
    }

    Ok(())
}
