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

fn terminal_cell(value: &str) -> String {
    value.replace(['\r', '\n'], " ").replace('`', "")
}

fn display_width(value: &str) -> usize {
    value.chars().count()
}

fn hard_wrap(value: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = value.chars().collect();
    chars.chunks(width).map(|chunk| chunk.iter().collect()).collect()
}

fn wrap_cell(value: &str, width: usize) -> Vec<String> {
    let value = terminal_cell(value);
    if value.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    for word in value.split_whitespace() {
        for part in hard_wrap(word, width) {
            if current.is_empty() {
                current = part;
            } else if display_width(&current) + 1 + display_width(&part) <= width {
                current.push(' ');
                current.push_str(&part);
            } else {
                lines.push(std::mem::take(&mut current));
                current = part;
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn table_border(widths: &[usize], left: char, join: char, right: char) -> String {
    let segments = widths
        .iter()
        .map(|width| "─".repeat(width + 2))
        .collect::<Vec<_>>()
        .join(&join.to_string());
    format!("{left}{segments}{right}\n")
}

fn push_table_row(
    output: &mut String,
    cells: &[String],
    widths: &[usize],
    right_aligned: &[usize],
) {
    let wrapped: Vec<Vec<String>> =
        cells.iter().enumerate().map(|(index, cell)| wrap_cell(cell, widths[index])).collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);

    for line_index in 0..height {
        output.push('│');
        for (column, width) in widths.iter().enumerate() {
            let cell = wrapped[column].get(line_index).map(String::as_str).unwrap_or("");
            let padding = width.saturating_sub(display_width(cell));
            output.push(' ');
            if right_aligned.contains(&column) {
                output.push_str(&" ".repeat(padding));
                output.push_str(cell);
            } else {
                output.push_str(cell);
                output.push_str(&" ".repeat(padding));
            }
            output.push(' ');
            output.push('│');
        }
        output.push('\n');
    }
}

fn render_table(
    headers: &[&str],
    rows: &[Vec<String>],
    max_widths: &[usize],
    right_aligned: &[usize],
) -> String {
    debug_assert_eq!(headers.len(), max_widths.len());
    debug_assert!(rows.iter().all(|row| row.len() == headers.len()));

    let header_cells: Vec<String> = headers.iter().map(|header| (*header).to_string()).collect();
    let mut widths: Vec<usize> = headers.iter().map(|header| display_width(header)).collect();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(display_width(&terminal_cell(cell)));
        }
    }
    for (index, width) in widths.iter_mut().enumerate() {
        *width = (*width).min(max_widths[index].max(display_width(headers[index])));
    }

    let mut output = table_border(&widths, '┌', '┬', '┐');
    push_table_row(&mut output, &header_cells, &widths, &[]);
    output.push_str(&table_border(&widths, '├', '┼', '┤'));
    for row in rows {
        push_table_row(&mut output, row, &widths, right_aligned);
    }
    output.push_str(&table_border(&widths, '└', '┴', '┘'));
    output
}

fn render_terminal(report: &serde_json::Value) -> String {
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

    writeln!(output, "Config").unwrap();
    output.push_str(&render_table(
        &["Field", "Value"],
        &[
            vec!["Path".into(), path.into()],
            vec!["Status".into(), status],
            vec!["Default profile".into(), default_profile.into()],
            vec!["User profiles".into(), config["user_profiles"].to_string()],
            vec!["Built-in profiles".into(), config["builtin_profiles"].to_string()],
        ],
        &[20, 92],
        &[],
    ));

    if let Some(warnings) = config["warnings"].as_array()
        && !warnings.is_empty()
    {
        writeln!(output, "\nConfig warnings").unwrap();
        let rows = warnings
            .iter()
            .filter_map(|value| value.as_str())
            .map(|warning| vec![warning.to_string()])
            .collect::<Vec<_>>();
        output.push_str(&render_table(&["Warning"], &rows, &[112], &[]));
    }

    let mut adapter_rows = Vec::new();
    let mut detail_rows = Vec::new();
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
        let id = row["id"].as_str().unwrap_or("").to_string();
        adapter_rows.push(vec![
            id.clone(),
            row["transport"].as_str().unwrap_or("").to_string(),
            row["stability"].as_str().unwrap_or("").to_string(),
            status,
        ]);
        detail_rows.push(vec![id, efforts, row["notes"].as_str().unwrap_or("").to_string()]);
    }
    writeln!(output, "\nAdapters").unwrap();
    output.push_str(&render_table(
        &["Adapter", "Transport", "Stability", "Status"],
        &adapter_rows,
        &[10, 14, 14, 64],
        &[],
    ));
    writeln!(output, "\nAdapter capabilities").unwrap();
    output.push_str(&render_table(
        &["Adapter", "Efforts", "Notes"],
        &detail_rows,
        &[10, 32, 64],
        &[],
    ));

    let mut profile_rows = Vec::new();
    for row in report["profiles"].as_array().into_iter().flatten() {
        let usable = row["usable"].as_bool().unwrap_or(false);
        let status = if usable { "✓ usable" } else { "✗ unavailable" };
        let source = if row["builtin"].as_bool() == Some(true) { "built-in" } else { "user" };
        profile_rows.push(vec![
            row["name"].as_str().unwrap_or("").to_string(),
            source.to_string(),
            row["strategy"].as_str().unwrap_or("").to_string(),
            status.to_string(),
            format!("{} / {}", row["ready_members"], row["members"]),
            row["quorum"].to_string(),
        ]);
    }
    writeln!(output, "\nProfiles").unwrap();
    output.push_str(&render_table(
        &["Profile", "Source", "Strategy", "Status", "Ready", "Quorum"],
        &profile_rows,
        &[16, 10, 16, 16, 10, 10],
        &[4, 5],
    ));

    let summary = &report["summary"];
    writeln!(output, "\nSummary").unwrap();
    output.push_str(&render_table(
        &["Metric", "Ready", "Total"],
        &[
            vec![
                "Adapters".into(),
                summary["adapters_ready"].to_string(),
                summary["adapters_total"].to_string(),
            ],
            vec![
                "Profiles".into(),
                summary["profiles_usable"].to_string(),
                summary["profiles_total"].to_string(),
            ],
        ],
        &[16, 10, 10],
        &[1, 2],
    ));

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

    print!("{}", render_terminal(&report));
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
    fn terminal_report_renders_aligned_tables() {
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

        let rendered = render_terminal(&report);
        assert!(rendered.contains("┌"));
        assert!(rendered.contains("│ Adapter │ Transport │ Stability │ Status  │"));
        assert!(rendered.contains("│ claude  │ command   │ stable    │ ✓ ready │"));
        assert!(rendered.contains("native | CLI"));
        assert!(rendered.contains("old | key"));
        assert!(rendered.contains("│ deep    │ user   │ judge    │ ✓ usable │ 1 / 1 │      1 │"));
        assert!(!rendered.contains("| Adapter |"));
        assert!(!rendered.contains("## Adapters"));
    }

    #[test]
    fn terminal_table_wraps_long_cells_inside_its_border() {
        let rendered = render_table(&["Name"], &[vec!["abcdefgh".to_string()]], &[4], &[]);
        assert!(rendered.contains("│ abcd │"));
        assert!(rendered.contains("│ efgh │"));
    }
}
