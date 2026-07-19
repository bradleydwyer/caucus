use std::path::PathBuf;
use std::time::Duration;

use caucus_core::strategy::debate::DebateConfig;
use caucus_core::{
    Candidate, ConsensusResult, ConsensusStrategy, MultiRoundDebate, OutputFormat, consensus,
};
use clap::Args;
use colored::Colorize;

use super::run::RunDeadline;
use super::{build_single_provider, council, default_models};

#[derive(Args)]
pub struct AskArgs {
    /// The question or prompt to send to models
    pub prompt: String,

    /// Comma-separated list of models to query (defaults to all configured providers)
    #[arg(
        short,
        long,
        value_delimiter = ',',
        conflicts_with = "profile",
        conflicts_with = "auto",
        conflicts_with = "config"
    )]
    pub models: Option<Vec<String>>,

    /// Use a council profile's exact member specs instead of API models
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Build a zero-key council from locally discovered adapters (ready
    /// Claude/Codex, experimental Kimi/opencode/Grok at default pins, local
    /// servers with a discoverable model). Never requires API keys.
    #[arg(long, conflicts_with = "profile")]
    pub auto: bool,

    /// Path to a caucus config file (TOML)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Verbose output (show model queries, strategy, agreement score on stderr)
    #[arg(short, long)]
    pub verbose: bool,

    /// Custom system prompt for model queries
    #[arg(long)]
    pub system: Option<String>,

    /// Consensus strategy to use (defaults to judge, or the profile's strategy)
    #[arg(short, long,
        value_parser = ["majority-vote", "weighted-vote", "judge", "debate", "debate-then-vote"],
        long_help = "Consensus strategy to use:\n\
        \n  majority-vote     Count-based voting with fuzzy string matching (no LLM needed)\
        \n  weighted-vote     Candidates weighted by confidence/model reputation (no LLM needed)\
        \n  judge             A separate LLM evaluates all candidates and synthesizes the best response\
        \n  debate            Multi-round debate where candidates refine positions over rounds\
        \n  debate-then-vote  Debate rounds followed by majority vote (hybrid)\
        \n\nDefaults to judge, or the profile's strategy when --profile/--auto is used.")]
    pub strategy: Option<String>,

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
    #[arg(long, default_value = "3", value_parser = super::parse_rounds)]
    pub rounds: usize,
}

pub async fn run(args: AskArgs) -> anyhow::Result<()> {
    let format: OutputFormat = args.format.parse()?;
    if args.profile.is_some() || args.auto || args.config.is_some() {
        return run_council(args, format).await;
    }
    run_legacy(args, format).await
}

/// Profile/auto path: exact providers from member specs, bounded fan-out
/// with deadlines and quorum, then the selected strategy via `resolve_multi`
/// so debate refines each participant independently.
async fn run_council(args: AskArgs, format: OutputFormat) -> anyhow::Result<()> {
    let verbose = args.verbose;
    let (_path, config) = council::load_config(args.config.as_deref())?;

    let council = if args.auto {
        let selection = council::auto_council(&config).await;
        for exclusion in &selection.exclusions {
            eprintln!("  {} {} excluded: {}", "·".dimmed(), exclusion.adapter, exclusion.reason);
        }
        selection.council
    } else {
        match config.resolve_profile(args.profile.as_deref()) {
            Ok(council) => council,
            Err(caucus_core::ConfigError::NoDefault) => config.resolve_profile(Some("deep"))?,
            Err(error) => return Err(error.into()),
        }
    };

    if council.members.is_empty() {
        anyhow::bail!(
            "auto council is empty — no ready local adapters found; run `caucus doctor` for details"
        );
    }

    let strategy_name = args.strategy.clone().unwrap_or_else(|| council.strategy.clone());
    // Reject a misspelled profile strategy before spending any provider
    // requests. The validated strategy is constructed again after fan-out.
    caucus_core::strategy_from_name(&strategy_name)?;
    let deadline = RunDeadline::new(council.deadline_secs);

    if verbose {
        eprintln!(
            "{} Council '{}' ({} member(s), quorum {}, strategy '{}')",
            "▶".green(),
            council.name.cyan(),
            council.members.len(),
            council.quorum,
            strategy_name.cyan(),
        );
        for member in &council.members {
            eprintln!("  {} {}", "·".dimmed(), member.to_string().yellow());
        }
    }

    let multi = council::build_council_provider(&council, &config)?;
    let requested_timeout = Duration::from_secs(
        council
            .request_timeout_secs
            .or(council.deadline_secs)
            .unwrap_or(caucus_core::DEFAULT_REQUEST_TIMEOUT.as_secs()),
    );
    let timeout = deadline.turn_timeout(requested_timeout)?;
    let report = deadline
        .wait(caucus_core::provider::fanout(
            &multi,
            &args.prompt,
            args.system.as_deref(),
            caucus_core::FanoutConfig { max_concurrency: 4, timeout, quorum: council.quorum },
        ))
        .await?;

    for warning in report.warnings() {
        eprintln!("  {} {}", "✗".red(), warning);
    }
    if !report.quorum_met() {
        anyhow::bail!(
            "quorum not met for council '{}': {}/{} required member(s) responded",
            council.name,
            report.successes.len(),
            report.quorum,
        );
    }

    let candidates: Vec<Candidate> = report
        .successes
        .iter()
        .map(|s| {
            Candidate::new(s.content.clone())
                .with_model(s.model.clone())
                .with_metadata("question", serde_json::json!(&args.prompt))
                .with_metadata("latency_ms", serde_json::json!(s.latency_ms))
                .with_metadata("transport", serde_json::json!(s.transport.to_string()))
        })
        .collect();

    // Single member shortcut: skip consensus, just print the response.
    if candidates.len() == 1 && strategy_name == "judge" {
        if verbose {
            eprintln!("{} Single member — returning response directly\n", "✓".green());
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

    // Judge/fallback LLM: the profile's designated judge when set, otherwise
    // the first provider that actually returned a candidate. This prevents a
    // tolerated fan-out failure from becoming a fatal judge call.
    let judge = if strategy_needs_llm(&strategy_name) {
        if council.judge.is_some() {
            Some(council::select_judge(&council, &multi, &config)?)
        } else {
            let name =
                candidates.first().and_then(|candidate| candidate.model.clone()).ok_or_else(
                    || anyhow::anyhow!("no successful council member available as judge"),
                )?;
            let provider = multi.get(&name).ok_or_else(|| {
                anyhow::anyhow!("successful council provider '{name}' disappeared")
            })?;
            Some((name, council::JudgeProvider::Borrowed(provider)))
        }
    } else {
        None
    };
    if verbose && let Some((name, _)) = &judge {
        eprintln!("  {} judge: {}", "·".dimmed(), name.yellow());
    }
    let judge_llm: Option<&dyn caucus_core::LlmProvider> =
        judge.as_ref().map(|(_, provider)| provider.get());

    if verbose {
        eprintln!(
            "{} Got {} candidate(s), running {}...",
            "▶".green(),
            candidates.len(),
            strategy_name.cyan(),
        );
    }

    // Always go through resolve_multi: the default impl falls back to
    // resolve, while debate uses each member's own provider.
    let result = if is_debate_strategy(&strategy_name) {
        let strategy = MultiRoundDebate::with_config(DebateConfig {
            max_rounds: args.rounds,
            ..Default::default()
        });
        deadline.wait(strategy.resolve_multi(&candidates, judge_llm, Some(&multi))).await??
    } else {
        let strategy = caucus_core::strategy_from_name(&strategy_name)?;
        deadline.wait(strategy.resolve_multi(&candidates, judge_llm, Some(&multi))).await??
    };

    if verbose {
        eprintln!(
            "{} Consensus reached (agreement: {:.0}%)\n",
            "✓".green(),
            result.agreement_score * 100.0,
        );
    }

    println!("{}", format.render(&result));
    Ok(())
}

/// Legacy path: API models keyed from environment variables.
async fn run_legacy(args: AskArgs, format: OutputFormat) -> anyhow::Result<()> {
    let strategy_name = args.strategy.clone().unwrap_or_else(|| "judge".to_string());
    let models = args.models.clone().unwrap_or_else(default_models);
    let verbose = args.verbose;

    if verbose {
        eprintln!(
            "{} Querying {} model(s) with strategy '{}'...",
            "▶".green(),
            models.len(),
            strategy_name.cyan(),
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
    if candidates.len() == 1 && strategy_name == "judge" {
        if verbose {
            eprintln!("{} Single model — returning response directly\n", "✓".green(),);
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
            strategy_name.cyan(),
        );
    }

    // Run consensus
    let judge_llm: Option<Box<dyn caucus_core::LlmProvider>> = if strategy_needs_llm(&strategy_name)
    {
        // Use the first model as the judge
        let judge_model = models.first().expect("no models configured");
        Some(build_single_provider(judge_model)?)
    } else {
        None
    };
    let participant_provider = if is_debate_strategy(&strategy_name) {
        Some(super::build_provider(&models)?)
    } else {
        None
    };

    let result = if is_debate_strategy(&strategy_name) {
        let strategy = MultiRoundDebate::with_config(DebateConfig {
            max_rounds: args.rounds,
            ..Default::default()
        });
        strategy
            .resolve_multi(&candidates, judge_llm.as_deref(), participant_provider.as_ref())
            .await?
    } else {
        consensus(&candidates, &strategy_name, judge_llm.as_deref()).await?
    };

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

fn is_debate_strategy(name: &str) -> bool {
    matches!(name, "debate" | "multi_round_debate" | "multi-round-debate")
}

fn strategy_needs_llm(name: &str) -> bool {
    matches!(
        name,
        "judge"
            | "judge_synthesis"
            | "judge-synthesis"
            | "debate"
            | "multi_round_debate"
            | "multi-round-debate"
            | "debate_then_vote"
            | "debate-then-vote"
    )
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: AskArgs,
    }

    fn parse(argv: &[&str]) -> Result<AskArgs, clap::Error> {
        TestCli::try_parse_from(argv).map(|cli| cli.args)
    }

    #[test]
    fn legacy_models_still_parse() {
        let args = parse(&["caucus", "what is 2+2", "-m", "gpt-5.2,claude-opus-4-6"]).unwrap();
        assert_eq!(args.prompt, "what is 2+2");
        assert_eq!(
            args.models.unwrap(),
            vec!["gpt-5.2".to_string(), "claude-opus-4-6".to_string()]
        );
        assert!(args.profile.is_none());
        assert!(!args.auto);
        assert!(args.strategy.is_none()); // legacy default applied at run time
    }

    #[test]
    fn profile_and_auto_parse() {
        let args = parse(&["caucus", "q", "--profile", "deep", "--config", "caucus.toml"]).unwrap();
        assert_eq!(args.profile.as_deref(), Some("deep"));
        assert_eq!(args.config, Some(PathBuf::from("caucus.toml")));

        let args = parse(&["caucus", "q", "--auto"]).unwrap();
        assert!(args.auto);
        assert!(args.profile.is_none());
    }

    #[test]
    fn profile_and_auto_conflict_with_models() {
        assert!(parse(&["caucus", "q", "-m", "gpt-5.2", "--profile", "deep"]).is_err());
        assert!(parse(&["caucus", "q", "-m", "gpt-5.2", "--auto"]).is_err());
        assert!(parse(&["caucus", "q", "-m", "gpt-5.2", "--config", "caucus.toml"]).is_err());
    }

    #[test]
    fn strategy_override_is_optional_and_validated() {
        let args = parse(&["caucus", "q", "--profile", "deep", "-s", "debate"]).unwrap();
        assert_eq!(args.strategy.as_deref(), Some("debate"));
        assert!(parse(&["caucus", "q", "-s", "bogus"]).is_err());
    }
}
