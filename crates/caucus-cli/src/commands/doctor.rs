//! `caucus doctor`: report adapter readiness, config validity, and profile
//! health. Probes the local machine only — never reads credentials.

use std::collections::BTreeMap;
use std::path::PathBuf;

use caucus_core::adapters::{self, AdapterDescriptor, Readiness};
use caucus_core::{Config, Transport, builtin_profiles};
use clap::Args;
use colored::Colorize;

use super::council;

#[derive(Args)]
pub struct DoctorArgs {
    /// Emit the full report as JSON
    #[arg(long)]
    pub json: bool,

    /// Path to a caucus config file (TOML)
    #[arg(long)]
    pub config: Option<PathBuf>,
}

/// One adapter's health row.
#[derive(Debug, Clone, serde::Serialize)]
struct AdapterRow {
    id: String,
    transport: String,
    stability: String,
    ready: bool,
    readiness: String,
    efforts: Vec<String>,
    notes: String,
}

/// One profile's health row.
#[derive(Debug, Clone, serde::Serialize)]
struct ProfileRow {
    name: String,
    builtin: bool,
    strategy: String,
    quorum: usize,
    members: usize,
    ready_members: usize,
    /// Enough ready members to satisfy quorum right now.
    usable: bool,
}

/// Readiness for one descriptor, honoring a `binary_path` override from the
/// loaded config (discovery itself only probes PATH).
fn readiness_for(
    d: &AdapterDescriptor,
    discovered: &Readiness,
    config: Option<&Config>,
) -> (bool, String) {
    if d.transport == Transport::Command
        && let Some(config) = config
        && let Ok(utility) = d.id.parse()
        && let Ok(typed) = council::adapter_config(config, utility)
        && let Some(path) = typed.binary_path
    {
        return if path.is_file() {
            (true, format!("ready (binary override {})", path.display()))
        } else {
            (false, format!("binary override {} does not exist", path.display()))
        };
    }
    (discovered.is_ready(), discovered.to_string())
}

/// Per-profile readiness against the set of ready adapter ids. Pure so it is
/// unit-testable without probing the machine.
fn profile_rows(config: &Config, ready: &BTreeMap<String, bool>) -> Vec<ProfileRow> {
    config
        .list_profiles()
        .into_iter()
        .filter_map(|(name, _profile)| {
            let council = config.resolve_profile(Some(&name)).ok()?;
            let ready_members = council
                .members
                .iter()
                .filter(|m| ready.get(m.utility.as_str()).copied().unwrap_or(false))
                .count();
            Some(ProfileRow {
                builtin: builtin_profiles().contains_key(&name)
                    && !config.profiles.contains_key(&name),
                strategy: council.strategy,
                quorum: council.quorum,
                members: council.members.len(),
                ready_members,
                usable: ready_members >= council.quorum,
                name,
            })
        })
        .collect()
}

/// Build the full machine-readable report.
async fn report(config_path: Option<PathBuf>, config: &Config) -> serde_json::Value {
    let discovered = adapters::discover().await;
    let mut ready: BTreeMap<String, bool> = BTreeMap::new();
    let rows: Vec<AdapterRow> = discovered
        .iter()
        .map(|d| {
            let (is_ready, label) = readiness_for(d.descriptor, &d.readiness, Some(config));
            ready.insert(d.descriptor.id.to_string(), is_ready);
            AdapterRow {
                id: d.descriptor.id.to_string(),
                transport: d.descriptor.transport.to_string(),
                stability: d.descriptor.stability.to_string(),
                ready: is_ready,
                readiness: label,
                efforts: d.descriptor.efforts.iter().map(|e| e.as_str().to_string()).collect(),
                notes: d.descriptor.notes.to_string(),
            }
        })
        .collect();

    let profiles = profile_rows(config, &ready);
    let usable_profiles = profiles.iter().filter(|p| p.usable).count();

    serde_json::json!({
        "config": {
            "path": config_path.map(|p| p.display().to_string()),
            "valid": true,
            "default_profile": config.default_profile,
            "user_profiles": config.profiles.len(),
            "builtin_profiles": builtin_profiles().len(),
            "warnings": config.warnings,
        },
        "adapters": rows,
        "profiles": profiles,
        "summary": {
            "adapters_ready": ready.values().filter(|v| **v).count(),
            "adapters_total": ready.len(),
            "profiles_usable": usable_profiles,
            "profiles_total": profiles.len(),
        },
    })
}

pub async fn run(args: DoctorArgs) -> anyhow::Result<()> {
    // An explicit-but-invalid config is reported, not fatal: doctor exists to
    // surface exactly that kind of problem.
    let (config_path, config, config_error) = match council::load_config(args.config.as_deref()) {
        Ok((path, config)) => (path, config, None),
        Err(e) => (args.config.clone(), Config::default(), Some(e.to_string())),
    };

    let mut report = report(config_path, &config).await;
    if let Some(error) = &config_error {
        report["config"]["valid"] = serde_json::json!(false);
        report["config"]["error"] = serde_json::json!(error);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    // Human-readable rendering.
    let config = &report["config"];
    let valid = config["valid"].as_bool() == Some(true);
    match (&config["path"], valid) {
        (serde_json::Value::String(path), _) => println!("{} {path}", "Config:".bold()),
        (_, true) => println!("{} none found (using defaults + built-ins)", "Config:".bold()),
        (_, false) => println!("{} discovery failed", "Config:".bold()),
    }
    if valid {
        let default = config["default_profile"].as_str().unwrap_or("none");
        println!(
            "  valid; default profile: {default}; user profiles: {}; built-ins: {}",
            config["user_profiles"], config["builtin_profiles"]
        );
    } else {
        println!("  {} {}", "✗".red(), config["error"].as_str().unwrap_or("invalid"));
    }

    println!("\n{}", "Adapters:".bold());
    for row in report["adapters"].as_array().into_iter().flatten() {
        let ready = row["ready"].as_bool().unwrap_or(false);
        let mark = if ready { "✓".green() } else { "✗".red() };
        let efforts = row["efforts"]
            .as_array()
            .map(|e| {
                if e.is_empty() {
                    "none".to_string()
                } else {
                    e.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", ")
                }
            })
            .unwrap_or_else(|| "none".to_string());
        println!(
            "  {mark} {:<9} {:<12} {:<12} {}",
            row["id"].as_str().unwrap_or(""),
            row["transport"].as_str().unwrap_or(""),
            row["stability"].as_str().unwrap_or(""),
            row["readiness"].as_str().unwrap_or(""),
        );
        println!("      efforts: {efforts}; {}", row["notes"].as_str().unwrap_or(""));
    }

    println!("\n{}", "Profiles:".bold());
    for row in report["profiles"].as_array().into_iter().flatten() {
        let usable = row["usable"].as_bool().unwrap_or(false);
        let mark = if usable { "✓".green() } else { "✗".yellow() };
        let kind = if row["builtin"].as_bool() == Some(true) { "built-in" } else { "user" };
        println!(
            "  {mark} {:<16} {:<8} strategy={} quorum={} ready={}/{}",
            row["name"].as_str().unwrap_or(""),
            kind,
            row["strategy"].as_str().unwrap_or(""),
            row["quorum"],
            row["ready_members"],
            row["members"],
        );
    }

    let summary = &report["summary"];
    println!(
        "\n{} {}/{} adapters ready, {}/{} profiles usable",
        "Summary:".bold(),
        summary["adapters_ready"],
        summary["adapters_total"],
        summary["profiles_usable"],
        summary["profiles_total"],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_rows_count_ready_members_against_quorum() {
        let config = Config::from_toml_str(
            r#"
[profiles.mixed]
strategy = "judge"
quorum = 1
members = ["claude:opus@high", "codex:default@high"]
"#,
        )
        .unwrap();
        let ready = BTreeMap::from([("claude".to_string(), true), ("codex".to_string(), false)]);
        let rows = profile_rows(&config, &ready);
        let mixed = rows.iter().find(|r| r.name == "mixed").unwrap();
        assert_eq!(mixed.members, 2);
        assert_eq!(mixed.ready_members, 1);
        assert!(mixed.usable); // quorum 1 is satisfied
        assert!(!mixed.builtin);

        // The built-in deep profile needs 3 ready members.
        let deep = rows.iter().find(|r| r.name == "deep").unwrap();
        assert!(deep.builtin);
        assert_eq!(deep.quorum, 3);
        assert!(!deep.usable);
    }

    #[test]
    fn user_profile_shadowing_builtin_is_not_marked_builtin() {
        let config = Config::from_toml_str(
            r#"
[profiles.deep]
quorum = 1
members = ["codex:default@high"]
"#,
        )
        .unwrap();
        let rows = profile_rows(&config, &BTreeMap::new());
        let deep = rows.iter().find(|r| r.name == "deep").unwrap();
        assert!(!deep.builtin);
    }

    #[test]
    fn invalid_profiles_are_skipped_not_fatal() {
        // resolve_profile failures (none expected post-validation, but be
        // defensive) must not break the report.
        let config = Config::default();
        let rows = profile_rows(&config, &BTreeMap::new());
        assert!(rows.iter().any(|r| r.name == "deep"));
    }
}
