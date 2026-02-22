use clap::Args;
use colored::Colorize;
use conroute_core::{consensus, Candidate, ConsensusResult, OutputFormat};

use super::{build_single_provider, default_models};

#[derive(Args)]
pub struct AskArgs {
    /// The question or prompt to send to models
    pub prompt: String,

    /// Comma-separated list of models to query (defaults to all configured providers)
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Verbose output (show model queries, strategy, agreement score on stderr)
    #[arg(short, long)]
    pub verbose: bool,

    /// Custom system prompt for model queries
    #[arg(long)]
    pub system: Option<String>,

    /// Consensus strategy to use
    #[arg(short, long, default_value = "judge",
        value_parser = ["majority-vote", "weighted-vote", "judge", "debate", "debate-then-vote"],
        long_help = "Consensus strategy to use:\n\
        \n  majority-vote     Count-based voting with fuzzy string matching (no LLM needed)\
        \n  weighted-vote     Candidates weighted by confidence/model reputation (no LLM needed)\
        \n  judge             A separate LLM evaluates all candidates and synthesizes the best response\
        \n  debate            Multi-round debate where candidates refine positions over rounds\
        \n  debate-then-vote  Debate rounds followed by majority vote (hybrid)")]
    pub strategy: String,

    /// Output format
    #[arg(short, long, default_value = "plain",
        value_parser = ["plain", "json", "supreme-court", "detailed"],
        long_help = "Output format:\n\
        \n  plain         Consensus response as text (default)\
        \n  json          Full ConsensusResult as JSON\
        \n  supreme-court Majority opinion + concurrences + dissents + vote summary\
        \n  detailed      Full transcript with all candidates, metadata, and process info")]
    pub format: String,

    /// Number of debate rounds (for debate strategies)
    #[arg(long, default_value = "3")]
    pub rounds: usize,
}

pub async fn run(args: AskArgs) -> anyhow::Result<()> {
    let format = OutputFormat::from_str(&args.format)?;
    let models = args.models.unwrap_or_else(default_models);
    let verbose = args.verbose;

    if verbose {
        eprintln!(
            "{} Querying {} model(s) with strategy '{}'...",
            "▶".green(),
            models.len(),
            args.strategy.cyan(),
        );
    }

    // Generate candidates from all models in parallel
    let prompt = args.prompt.clone();
    let system = args.system.clone();
    let mut handles = Vec::new();

    for model in &models {
        let llm = build_single_provider(model)?;
        let model = model.clone();
        let prompt = prompt.clone();
        let system = system.clone();

        if verbose {
            eprintln!("  {} Querying {}...", "·".dimmed(), model.yellow());
        }

        handles.push(tokio::spawn(async move {
            let result = llm.complete(&prompt, system.as_deref()).await;
            (model, result)
        }));
    }

    let mut candidates = Vec::new();
    for handle in handles {
        let (model, result) = handle.await?;
        match result {
            Ok(response) => {
                candidates.push(
                    Candidate::new(response)
                        .with_model(model)
                        .with_metadata("question", serde_json::json!(&prompt)),
                );
            }
            Err(e) => {
                eprintln!("  {} {} failed: {}", "✗".red(), model, e);
            }
        }
    }

    if candidates.is_empty() {
        anyhow::bail!("No candidates generated — all models failed");
    }

    // Single model shortcut: skip consensus, just print the response directly
    if candidates.len() == 1 && args.strategy == "judge" {
        if verbose {
            eprintln!(
                "{} Single model — returning response directly\n",
                "✓".green(),
            );
        }
        let result = ConsensusResult {
            content: candidates[0].content.clone(),
            strategy: "passthrough".into(),
            agreement_score: 1.0,
            candidates,
            dissents: vec![],
            reasoning: None,
            metadata: Default::default(),
        };
        println!("{}", format.render(&result));
        return Ok(());
    }

    if verbose {
        eprintln!(
            "{} Got {} candidate(s), running {}...",
            "▶".green(),
            candidates.len(),
            args.strategy.cyan(),
        );
    }

    // Run consensus
    let judge_llm: Option<Box<dyn conroute_core::LlmProvider>> =
        if strategy_needs_llm(&args.strategy) {
            // Use the first model as the judge
            let judge_model = models.first().unwrap();
            Some(build_single_provider(judge_model)?)
        } else {
            None
        };

    let result = consensus(
        &candidates,
        &args.strategy,
        judge_llm.as_deref(),
    )
    .await?;

    if verbose {
        eprintln!(
            "{} Consensus reached (agreement: {:.0}%)\n",
            "✓".green(),
            result.agreement_score * 100.0,
        );
    }

    // Output
    println!("{}", format.render(&result));

    Ok(())
}

fn strategy_needs_llm(name: &str) -> bool {
    matches!(
        name,
        "judge" | "judge_synthesis" | "judge-synthesis"
            | "debate" | "multi_round_debate" | "multi-round-debate"
            | "debate_then_vote" | "debate-then-vote"
    )
}
