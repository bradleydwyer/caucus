//! `caucus doctor`: report adapter readiness, config validity, and profile
//! health. Probes the local machine only — never reads credentials.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use caucus_core::adapters::{self, AdapterDescriptor, Readiness};
use caucus_core::{Config, Transport, builtin_profiles};
use clap::Args;

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

fn markdown_cell(value: &str) -> String {
    value.replace('\\', "\\\\").replace('|', "\\|").replace(['\r', '\n'], "<br>")
}

fn render_markdown(report: &serde_json::Value) -> String {
    let mut output = String::new();
    let config = &report["config"];
    let valid = config["valid"].as_bool() == Some(true);
    let path = config["path"].as_str().unwrap_or(if valid {
        "none found (using defaults + built-ins)"
    } else {
        "discovery failed"
    });
    let status = if valid {
        "✓ Valid".to_string()
    } else {
        format!("✗ {}", config["error"].as_str().unwrap_or("invalid"))
    };
    let default_profile = config["default_profile"].as_str().unwrap_or("none");

    writeln!(output, "## Config\n").unwrap();
    writeln!(output, "| Field | Value |").unwrap();
    writeln!(output, "| :--- | :--- |").unwrap();
    writeln!(output, "| Path | {} |", markdown_cell(path)).unwrap();
    writeln!(output, "| Status | {} |", markdown_cell(&status)).unwrap();
    writeln!(output, "| Default profile | {} |", markdown_cell(default_profile)).unwrap();
    writeln!(output, "| User profiles | {} |", config["user_profiles"]).unwrap();
    writeln!(output, "| Built-in profiles | {} |", config["builtin_profiles"]).unwrap();

    if let Some(warnings) = config["warnings"].as_array()
        && !warnings.is_empty()
    {
        writeln!(output, "\n## Config warnings\n").unwrap();
        writeln!(output, "| Warning |").unwrap();
        writeln!(output, "| :--- |").unwrap();
        for warning in warnings.iter().filter_map(|value| value.as_str()) {
            writeln!(output, "| {} |", markdown_cell(warning)).unwrap();
        }
    }

    writeln!(output, "\n## Adapters\n").unwrap();
    writeln!(output, "| Adapter | Transport | Stability | Status | Efforts | Notes |").unwrap();
    writeln!(output, "| :--- | :--- | :--- | :--- | :--- | :--- |").unwrap();
    for row in report["adapters"].as_array().into_iter().flatten() {
        let ready = row["ready"].as_bool().unwrap_or(false);
        let readiness = row["readiness"].as_str().unwrap_or("");
        let status = format!("{} {readiness}", if ready { "✓" } else { "✗" });
        let efforts = row["efforts"]
            .as_array()
            .map(|items| {
                if items.is_empty() {
                    "none".to_string()
                } else {
                    items.iter().filter_map(|item| item.as_str()).collect::<Vec<_>>().join(", ")
                }
            })
            .unwrap_or_else(|| "none".to_string());
        writeln!(
            output,
            "| {} | {} | {} | {} | {} | {} |",
            markdown_cell(row["id"].as_str().unwrap_or("")),
            markdown_cell(row["transport"].as_str().unwrap_or("")),
            markdown_cell(row["stability"].as_str().unwrap_or("")),
            markdown_cell(&status),
            markdown_cell(&efforts),
            markdown_cell(row["notes"].as_str().unwrap_or("")),
        )
        .unwrap();
    }

    writeln!(output, "\n## Profiles\n").unwrap();
    writeln!(output, "| Profile | Source | Strategy | Status | Ready | Quorum |").unwrap();
    writeln!(output, "| :--- | :--- | :--- | :--- | ---: | ---: |").unwrap();
    for row in report["profiles"].as_array().into_iter().flatten() {
        let usable = row["usable"].as_bool().unwrap_or(false);
        let status = if usable { "✓ usable" } else { "✗ unavailable" };
        let source = if row["builtin"].as_bool() == Some(true) { "built-in" } else { "user" };
        writeln!(
            output,
            "| {} | {source} | {} | {status} | {} / {} | {} |",
            markdown_cell(row["name"].as_str().unwrap_or("")),
            markdown_cell(row["strategy"].as_str().unwrap_or("")),
            row["ready_members"],
            row["members"],
            row["quorum"],
        )
        .unwrap();
    }

    let summary = &report["summary"];
    writeln!(output, "\n## Summary\n").unwrap();
    writeln!(output, "| Metric | Ready | Total |").unwrap();
    writeln!(output, "| :--- | ---: | ---: |").unwrap();
    writeln!(
        output,
        "| Adapters | {} | {} |",
        summary["adapters_ready"], summary["adapters_total"]
    )
    .unwrap();
    writeln!(
        output,
        "| Profiles | {} | {} |",
        summary["profiles_usable"], summary["profiles_total"]
    )
    .unwrap();

    output
}

pub async fn run(args: DoctorArgs) -> anyhow::Result<()> {
    // An explicit-but-invalid config is reported, not fatal: doctor exists to
    // surface exactly that kind of problem.
    let (config_path, config, config_error) =
        match council::load_config_quiet(args.config.as_deref()) {
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

    print!("{}", render_markdown(&report));
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

    #[test]
    fn markdown_report_uses_tables_and_escapes_cells() {
        let report = serde_json::json!({
            "config": {
                "path": "/tmp/caucus.toml",
                "valid": true,
                "default_profile": "deep",
                "user_profiles": 1,
                "builtin_profiles": 1,
                "warnings": ["old | key"],
            },
            "adapters": [{
                "id": "claude",
                "transport": "command",
                "stability": "stable",
                "ready": true,
                "readiness": "ready",
                "efforts": ["high", "xhigh"],
                "notes": "native | CLI",
            }],
            "profiles": [{
                "name": "deep",
                "builtin": false,
                "strategy": "judge",
                "quorum": 1,
                "members": 1,
                "ready_members": 1,
                "usable": true,
            }],
            "summary": {
                "adapters_ready": 1,
                "adapters_total": 1,
                "profiles_usable": 1,
                "profiles_total": 1,
            },
        });

        let rendered = render_markdown(&report);
        assert!(
            rendered.contains("| Adapter | Transport | Stability | Status | Efforts | Notes |")
        );
        assert!(
            rendered
                .contains("| claude | command | stable | ✓ ready | high, xhigh | native \\| CLI |")
        );
        assert!(rendered.contains("| old \\| key |"));
        assert!(rendered.contains("| deep | user | judge | ✓ usable | 1 / 1 | 1 |"));
        assert!(rendered.contains("| Adapters | 1 | 1 |"));
    }
}
