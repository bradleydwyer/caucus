mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Check if we need to inject "ask" as the default subcommand.
/// Returns true when the first positional arg is not a known subcommand.
fn needs_subcommand_injection(args: &[String]) -> bool {
    let subcommands = ["ask", "compare", "debate", "bench", "serve", "help"];

    // Find the first positional argument by skipping global flags
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--help" | "-h" | "--version" | "-V" => return false,
            "--env" => { i += 2; continue; }         // --env <path>
            _ if arg.starts_with("--env=") => { i += 1; continue; }
            _ if arg.starts_with('-') => { i += 1; continue; } // skip subcommand flags
            _ => return !subcommands.contains(&arg),  // first positional arg
        }
    }
    false // no positional args → show help
}

#[derive(Parser)]
#[command(name = "conroute")]
#[command(about = "Multi-LLM consensus engine — aggregate and synthesize LLM outputs")]
#[command(version)]
struct Cli {
    /// Path to .env file to load
    #[arg(long, global = true)]
    env: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// One-shot consensus query across multiple models
    Ask(commands::ask::AskArgs),
    /// Compare multiple strategies side-by-side
    Compare(commands::compare::CompareArgs),
    /// Interactive multi-round debate between models
    Debate(commands::debate::DebateArgs),
    /// Batch evaluation across a dataset
    Bench(commands::bench::BenchArgs),
    /// Start an HTTP API or MCP server
    Serve(commands::serve::ServeArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("conroute=info".parse()?),
        )
        .init();

    // If no subcommand is given, treat it as `ask` (e.g. `conroute "question"`)
    let mut args: Vec<String> = std::env::args().collect();
    if needs_subcommand_injection(&args) {
        args.insert(1, "ask".into());
    }
    let cli = Cli::parse_from(args);

    // Load .env file: explicit --env path, or auto-discover .env in cwd
    if let Some(env_path) = &cli.env {
        dotenvy::from_path(env_path)
            .map_err(|e| anyhow::anyhow!("Failed to load env file {:?}: {e}", env_path))?;
    } else {
        let _ = dotenvy::dotenv(); // silently ignore if no .env found
    }

    match cli.command {
        Commands::Ask(args) => commands::ask::run(args).await,
        Commands::Compare(args) => commands::compare::run(args).await,
        Commands::Debate(args) => commands::debate::run(args).await,
        Commands::Bench(args) => commands::bench::run(args).await,
        Commands::Serve(args) => commands::serve::run(args).await,
    }
}
