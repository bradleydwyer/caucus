use std::path::PathBuf;

use clap::Args;
use colored::Colorize;
use caucus_core::{consensus, Candidate};

use super::{build_provider, build_single_provider, default_models};

#[derive(Args)]
pub struct BenchArgs {
    /// Path to JSONL file with test cases (each line: {"prompt": "...", "expected": "..."})
    pub input: PathBuf,

    /// Comma-separated list of models to query (defaults to all configured providers)
    #[arg(short, long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Comma-separated list of strategies to benchmark
    #[arg(short, long, value_delimiter = ',', default_value = "majority-vote",
        value_parser = ["majority-vote", "weighted-vote", "judge", "debate", "debate-then-vote"])]
    pub strategies: Vec<String>,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(serde::Deserialize)]
struct TestCase {
    prompt: String,
    #[serde(default)]
    expected: Option<String>,
}

#[derive(serde::Serialize)]
struct BenchResult {
    prompt: String,
    strategy: String,
    content: String,
    agreement_score: f64,
    expected: Option<String>,
    matched: Option<bool>,
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    let input = std::fs::read_to_string(&args.input)?;
    let test_cases: Vec<TestCase> = input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()?;

    let models = args.models.unwrap_or_else(default_models);

    eprintln!(
        "{} Running {} test cases across {} strategies and {} models",
        "▶".green(),
        test_cases.len(),
        args.strategies.len(),
        models.len(),
    );

    let provider = build_provider(&models)?;
    let judge_llm: Option<Box<dyn caucus_core::LlmProvider>> = {
        let judge_model = models.first().unwrap();
        build_single_provider(judge_model).ok()
    };

    let mut all_results = Vec::new();

    for (i, test_case) in test_cases.iter().enumerate() {
        eprintln!(
            "\n{} Test case {}/{}: {:?}",
            "▶".green(),
            i + 1,
            test_cases.len(),
            &test_case.prompt[..test_case.prompt.len().min(60)],
        );

        // Generate candidates
        let mut candidates = Vec::new();
        for model in &models {
            let llm = provider.get(model).unwrap();
            match llm.complete(&test_case.prompt, None).await {
                Ok(response) => {
                    candidates.push(Candidate::new(response).with_model(model.clone()));
                }
                Err(e) => {
                    eprintln!("  {} {} failed: {}", "✗".red(), model, e);
                }
            }
        }

        if candidates.is_empty() {
            continue;
        }

        // Run each strategy
        for strategy_name in &args.strategies {
            let result = consensus(&candidates, strategy_name, judge_llm.as_deref()).await;

            match result {
                Ok(result) => {
                    let matched = test_case.expected.as_ref().map(|exp| {
                        result.content.to_lowercase().contains(&exp.to_lowercase())
                    });

                    let status = match matched {
                        Some(true) => "✓".green(),
                        Some(false) => "✗".red(),
                        None => "·".dimmed(),
                    };

                    eprintln!(
                        "  {} {} — agreement: {:.0}%",
                        status,
                        strategy_name,
                        result.agreement_score * 100.0,
                    );

                    all_results.push(BenchResult {
                        prompt: test_case.prompt.clone(),
                        strategy: strategy_name.clone(),
                        content: result.content,
                        agreement_score: result.agreement_score,
                        expected: test_case.expected.clone(),
                        matched,
                    });
                }
                Err(e) => {
                    eprintln!("  {} {} failed: {}", "✗".red(), strategy_name, e);
                }
            }
        }
    }

    // Output results
    let json = serde_json::to_string_pretty(&all_results)?;

    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &json)?;
        eprintln!("\n{} Results written to {:?}", "✓".green(), output_path);
    } else {
        println!("{json}");
    }

    // Print summary
    let total = all_results.len();
    let matched = all_results.iter().filter(|r| r.matched == Some(true)).count();
    let unmatched = all_results.iter().filter(|r| r.matched == Some(false)).count();
    let no_expected = all_results.iter().filter(|r| r.matched.is_none()).count();
    let avg_agreement: f64 = if total > 0 {
        all_results.iter().map(|r| r.agreement_score).sum::<f64>() / total as f64
    } else {
        0.0
    };

    eprintln!("\n{}", "═══ Summary ═══".bold());
    eprintln!("Total:           {total}");
    eprintln!("Matched:         {}", matched.to_string().green());
    eprintln!("Unmatched:       {}", unmatched.to_string().red());
    eprintln!("No expectation:  {no_expected}");
    eprintln!("Avg agreement:   {:.0}%", avg_agreement * 100.0);

    Ok(())
}
