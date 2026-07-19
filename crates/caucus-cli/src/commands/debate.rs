use std::io::Write;

use caucus_core::strategy::debate::DebateConfig;
use caucus_core::{
    Candidate, ConsensusStrategy, DebateEvent, FanoutConfig, MultiRoundDebate, OutputFormat,
};
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

    /// Stream initial positions and each completed round response to stderr
    #[arg(long)]
    pub live: bool,

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
    let result = if args.live {
        strategy
            .resolve_multi_observed(
                &candidates,
                Some(judge_llm),
                Some(&provider),
                &print_live_event,
            )
            .await?
    } else {
        strategy.resolve_multi(&candidates, Some(judge_llm), Some(&provider)).await?
    };

    eprintln!(
        "\n{} Debate concluded (agreement: {:.0}%)\n",
        "✓".green(),
        result.agreement_score * 100.0,
    );

    println!("{}", format.render(&result));

    Ok(())
}

fn participant_label(participant: usize, model: Option<&str>) -> String {
    model.map(str::to_string).unwrap_or_else(|| format!("participant {}", participant + 1))
}

fn render_live_event(event: &DebateEvent) -> String {
    match event {
        DebateEvent::InitialPosition { participant, model, content } => format!(
            "── Initial position · {} ──\n{}",
            participant_label(*participant, model.as_deref()),
            content
        ),
        DebateEvent::RoundStarted { round, max_rounds } => {
            format!("▶ Round {round}/{max_rounds}")
        }
        DebateEvent::PositionRevised { round, participant, model, response, .. } => format!(
            "── Round {round} · {} ──\n{}",
            participant_label(*participant, model.as_deref()),
            response
        ),
        DebateEvent::PositionRetained { round, participant, model, reason } => format!(
            "✗ Round {round} · {} kept its previous position ({reason})",
            participant_label(*participant, model.as_deref())
        ),
        DebateEvent::RoundCompleted { round, mean_similarity: Some(similarity) } => {
            format!(
                "✓ Round {round} complete (mean position similarity: {:.0}%)",
                similarity * 100.0
            )
        }
        DebateEvent::RoundCompleted { round, mean_similarity: None } => {
            format!("✓ Round {round} complete (no successful revisions)")
        }
        DebateEvent::Converged { round, mean_similarity } => format!(
            "✓ Debate converged after round {round} ({:.0}% similarity)",
            mean_similarity * 100.0
        ),
        DebateEvent::AdjudicationStarted => "▶ Adjudicating final positions".to_string(),
    }
}

fn print_live_event(event: &DebateEvent) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "\n{}", render_live_event(event));
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: DebateArgs,
    }

    #[test]
    fn live_flag_is_opt_in() {
        let normal = TestCli::try_parse_from(["test", "topic"]).unwrap();
        assert!(!normal.args.live);

        let live = TestCli::try_parse_from(["test", "topic", "--live"]).unwrap();
        assert!(live.args.live);
        assert_eq!(live.args.format, "detailed");
    }

    #[test]
    fn live_revision_includes_model_round_and_full_response() {
        let rendered = render_live_event(&DebateEvent::PositionRevised {
            round: 2,
            participant: 0,
            model: Some("claude-opus".to_string()),
            response: "Critique first.\nFINAL ANSWER: Revised view.".to_string(),
            position: "Revised view.".to_string(),
        });

        assert!(rendered.contains("Round 2 · claude-opus"));
        assert!(rendered.contains("Critique first."));
        assert!(rendered.contains("FINAL ANSWER: Revised view."));
    }
}
