//! `caucus profiles [NAME]`: list or show council profiles — built-ins plus
//! user profiles — with exact `utility:model@effort` member strings.

use std::path::PathBuf;

use caucus_core::{Config, builtin_profiles};
use clap::Args;
use colored::Colorize;

use super::council;

#[derive(Args)]
pub struct ProfilesArgs {
    /// Show a single profile in detail (name or the configured default)
    pub name: Option<String>,

    /// Emit the listing as JSON
    #[arg(long)]
    pub json: bool,

    /// Path to a caucus config file (TOML)
    #[arg(long)]
    pub config: Option<PathBuf>,
}

/// One profile in the listing, with resolved exact member strings.
#[derive(Debug, Clone, serde::Serialize)]
struct ProfileEntry {
    name: String,
    builtin: bool,
    description: Option<String>,
    strategy: String,
    quorum: usize,
    deadline_secs: Option<u64>,
    request_timeout_secs: Option<u64>,
    budget_usd: Option<f64>,
    judge: Option<String>,
    members: Vec<String>,
}

/// Resolve every listable profile (user profiles shadow built-ins). Pure so
/// it is unit-testable.
fn entries(config: &Config) -> Vec<ProfileEntry> {
    config
        .list_profiles()
        .into_iter()
        .filter_map(|(name, _)| {
            let council = config.resolve_profile(Some(&name)).ok()?;
            Some(ProfileEntry {
                builtin: builtin_profiles().contains_key(&name)
                    && !config.profiles.contains_key(&name),
                description: council.description,
                strategy: council.strategy,
                quorum: council.quorum,
                deadline_secs: council.deadline_secs,
                request_timeout_secs: council.request_timeout_secs,
                budget_usd: council.budget_usd,
                judge: council.judge.as_ref().map(ToString::to_string),
                members: council.members.iter().map(|m| m.to_string()).collect(),
                name,
            })
        })
        .collect()
}

fn render_entry(entry: &ProfileEntry) {
    let kind = if entry.builtin { "built-in".dimmed() } else { "user".cyan() };
    println!("{} {entry_name} [{kind}]", "●".green(), entry_name = entry.name.bold());
    if let Some(description) = &entry.description {
        println!("  {description}");
    }
    print!("  strategy={} quorum={}", entry.strategy, entry.quorum);
    if let Some(deadline) = entry.deadline_secs {
        print!(" deadline={deadline}s");
    }
    if let Some(timeout) = entry.request_timeout_secs {
        print!(" request-timeout={timeout}s");
    }
    if let Some(budget) = entry.budget_usd {
        print!(" budget=${budget:.2}");
    }
    if let Some(judge) = &entry.judge {
        print!(" judge={judge}");
    }
    println!();
    for member in &entry.members {
        println!("    - {member}");
    }
}

pub async fn run(args: ProfilesArgs) -> anyhow::Result<()> {
    let (_path, config) = council::load_config(args.config.as_deref())?;
    let all = entries(&config);

    if let Some(name) = &args.name {
        // `resolve_profile` gives the precise error for unknown names.
        config.resolve_profile(Some(name))?;
        let entry = all.into_iter().find(|e| e.name == *name).expect("resolved above");
        if args.json {
            println!("{}", serde_json::to_string_pretty(&entry)?);
        } else {
            render_entry(&entry);
        }
        return Ok(());
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&all)?);
        return Ok(());
    }

    if let Some(default) = &config.default_profile {
        println!("{} {default}\n", "Default profile:".bold());
    }
    for (i, entry) in all.iter().enumerate() {
        if i > 0 {
            println!();
        }
        render_entry(entry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_have_exact_member_strings() {
        let config = Config::from_toml_str(
            r#"
default_profile = "frontier"

[profiles.frontier]
description = "exact pins"
strategy = "judge"
quorum = 2
deadline_secs = 300
members = ["claude:claude-opus-4-8@xhigh", "kimi:kimi-code/k3@high"]
"#,
        )
        .unwrap();
        let all = entries(&config);
        let frontier = all.iter().find(|e| e.name == "frontier").unwrap();
        assert_eq!(
            frontier.members,
            vec!["claude:claude-opus-4-8@xhigh".to_string(), "kimi:kimi-code/k3@high".to_string()]
        );
        assert!(!frontier.builtin);
        assert_eq!(frontier.strategy, "judge");
        assert_eq!(frontier.quorum, 2);
        assert_eq!(frontier.deadline_secs, Some(300));

        // Built-in deep is present with its exact pins.
        let deep = all.iter().find(|e| e.name == "deep").unwrap();
        assert!(deep.builtin);
        assert_eq!(deep.members.len(), 6);
        assert_eq!(deep.members[0], "claude:opus@xhigh");
        assert_eq!(deep.members[5], "grok:grok-4.5@high");
    }

    #[test]
    fn entries_serialize_to_json() {
        let config = Config::default();
        let json = serde_json::to_value(entries(&config)).unwrap();
        let deep = json.as_array().unwrap().iter().find(|e| e["name"] == "deep").unwrap();
        assert_eq!(deep["builtin"], true);
        assert_eq!(deep["members"][0], "claude:opus@xhigh");
    }
}
