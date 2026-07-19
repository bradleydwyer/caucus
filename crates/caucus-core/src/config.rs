//! Council configuration: user profiles, built-ins, discovery, resolution.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub use crate::adapters::MemberSpec;

/// TOML shape of a single profile under `[profiles.<name>]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub description: Option<String>,
    pub strategy: Option<String>,
    pub quorum: Option<usize>,
    /// Whole-run wall-clock budget.
    pub deadline_secs: Option<u64>,
    /// Per-provider request/process timeout. When omitted, `deadline_secs`
    /// is also the per-request ceiling, then the library default applies.
    pub request_timeout_secs: Option<u64>,
    pub budget_usd: Option<f64>,
    /// Designated judge member (`utility:model@effort`) for judge/debate
    /// strategies. When unset, the first council member judges.
    pub judge: Option<String>,
    #[serde(default)]
    pub members: Vec<String>,
}

/// A profile with member strings validated into [`MemberSpec`]s.
#[derive(Debug, Clone)]
pub struct Council {
    pub name: String,
    pub description: Option<String>,
    pub members: Vec<MemberSpec>,
    /// Designated judge, validated like a member. `None` means the first
    /// member judges.
    pub judge: Option<MemberSpec>,
    pub strategy: String,
    pub quorum: usize,
    /// Whole-run wall-clock budget.
    pub deadline_secs: Option<u64>,
    /// Per-provider request/process timeout.
    pub request_timeout_secs: Option<u64>,
    pub budget_usd: Option<f64>,
}

/// Errors produced while loading, validating, or resolving configuration.
#[derive(Debug)]
pub enum ConfigError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { source: toml::de::Error },
    Invalid { message: String },
    UnknownProfile { name: String },
    NoDefault,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::Parse { source } => write!(f, "invalid TOML: {source}"),
            Self::Invalid { message } => write!(f, "invalid config: {message}"),
            Self::UnknownProfile { name } => write!(f, "unknown profile: {name}"),
            Self::NoDefault => write!(f, "no profile given and no default_profile set"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Loaded caucus configuration.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub default_profile: Option<String>,
    pub profiles: BTreeMap<String, Profile>,
    /// Per-adapter overrides from `[adapters.<name>]`.
    pub adapters: BTreeMap<String, toml::Value>,
    /// Warnings produced by the legacy-schema migration while parsing.
    /// Empty for files that use only the current schema.
    pub warnings: Vec<String>,
}

/// Raw TOML document before profiles are separated from adapter overrides.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    default_profile: Option<String>,
    #[serde(default)]
    profiles: BTreeMap<String, Profile>,
    #[serde(default)]
    adapters: BTreeMap<String, toml::Value>,
}

/// Built-in `deep` profile. User profiles shadow built-ins by name.
const BUILTIN_DEEP_TOML: &str = r#"
description = "Broad frontier panel judged by a chair"
strategy = "judge"
quorum = 3
deadline_secs = 600
members = [
    "claude:opus@xhigh",
    "claude:claude-fable-5@xhigh",
    "codex:default@xhigh",
    "opencode:zai-coding-plan/glm-5.2@xhigh",
    "kimi:kimi-code/k3@high",
    "grok:grok-4.5@high",
]
"#;

/// Built-in profiles, shadowed by user profiles of the same name.
pub fn builtin_profiles() -> BTreeMap<String, Profile> {
    let deep: Profile =
        toml::from_str(BUILTIN_DEEP_TOML).expect("built-in deep profile must parse");
    BTreeMap::from([("deep".to_string(), deep)])
}

impl Config {
    /// Parse a config from a TOML string and validate it.
    ///
    /// Legacy profile keys (`models`, `timeout_seconds`, `deadline_seconds`)
    /// are migrated in memory with recorded [`Config::warnings`]; the file
    /// on disk is never modified.
    pub fn from_toml_str(input: &str) -> Result<Self, ConfigError> {
        let mut doc: toml::Value =
            toml::from_str(input).map_err(|source| ConfigError::Parse { source })?;
        let mut warnings = Vec::new();
        migrate_legacy_profiles(&mut doc, &mut warnings);
        let raw: RawConfig = doc.try_into().map_err(|source| ConfigError::Parse { source })?;
        let config = Self {
            default_profile: raw.default_profile,
            profiles: raw.profiles,
            adapters: raw.adapters,
            warnings,
        };
        config.validate()?;
        Ok(config)
    }

    /// Load a config from an explicit path. Returns the path and the config.
    pub fn load(path: &Path) -> Result<(PathBuf, Self), ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|source| ConfigError::Io { path: path.to_path_buf(), source })?;
        Ok((path.to_path_buf(), Self::from_toml_str(&text)?))
    }

    /// Discover a config file. Search order: explicit path, `./caucus.toml`,
    /// `$XDG_CONFIG_HOME/caucus/config.toml`, `$HOME/.config/caucus/config.toml`.
    ///
    /// An explicit path that does not exist is an error; otherwise a missing
    /// file simply falls through to the next candidate. Returns `Ok(None)`
    /// when no candidate exists.
    pub fn discover(explicit: Option<&Path>) -> Result<Option<(PathBuf, Self)>, ConfigError> {
        let cwd = env::current_dir()
            .map_err(|source| ConfigError::Io { path: PathBuf::from("."), source })?;
        let xdg = env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        let home = env::var_os("HOME").map(PathBuf::from);
        Self::discover_in(explicit, &cwd, xdg.as_deref(), home.as_deref())
    }

    fn discover_in(
        explicit: Option<&Path>,
        cwd: &Path,
        xdg: Option<&Path>,
        home: Option<&Path>,
    ) -> Result<Option<(PathBuf, Self)>, ConfigError> {
        if let Some(path) = explicit {
            return Self::load(path).map(Some);
        }
        let mut candidates = vec![cwd.join("caucus.toml")];
        if let Some(xdg) = xdg {
            candidates.push(xdg.join("caucus").join("config.toml"));
        }
        if let Some(home) = home {
            candidates.push(home.join(".config").join("caucus").join("config.toml"));
        }
        for path in candidates {
            if path.is_file() {
                return Self::load(&path).map(Some);
            }
        }
        Ok(None)
    }

    /// Check every user profile: members parse, quorum is nonzero and no
    /// greater than the member count.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, profile) in &self.profiles {
            resolve(name, profile)?;
        }
        if let Some(default) = &self.default_profile
            && !self.profiles.contains_key(default)
            && !builtin_profiles().contains_key(default)
        {
            return Err(ConfigError::UnknownProfile { name: default.clone() });
        }
        Ok(())
    }

    /// Resolve a profile by name (or the configured default) into a
    /// [`Council`]. User profiles shadow built-ins of the same name.
    pub fn resolve_profile(&self, name: Option<&str>) -> Result<Council, ConfigError> {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.default_profile.clone().ok_or(ConfigError::NoDefault)?,
        };
        if let Some(profile) = self.profiles.get(&name) {
            return resolve(&name, profile);
        }
        if let Some(profile) = builtin_profiles().get(&name) {
            return resolve(&name, profile);
        }
        Err(ConfigError::UnknownProfile { name })
    }

    /// All resolvable profile names, user profiles first, then built-ins.
    pub fn profile_names(&self) -> Vec<String> {
        self.list_profiles().into_iter().map(|(name, _)| name).collect()
    }

    /// All resolvable profiles; user profiles shadow built-ins by name.
    pub fn list_profiles(&self) -> Vec<(String, Profile)> {
        let mut merged = builtin_profiles();
        for (name, profile) in &self.profiles {
            merged.insert(name.clone(), profile.clone());
        }
        merged.into_iter().collect()
    }

    /// Adapter-specific overrides from `[adapters.<name>]`, if any.
    pub fn adapter_config(&self, name: &str) -> Option<&toml::Value> {
        self.adapters.get(name)
    }
}

/// Validate a profile and build its [`Council`].
fn resolve(name: &str, profile: &Profile) -> Result<Council, ConfigError> {
    let members: Vec<MemberSpec> = profile
        .members
        .iter()
        .map(|raw| {
            raw.parse().map_err(|_| ConfigError::Invalid {
                message: format!("profile `{name}`: cannot parse member `{raw}`"),
            })
        })
        .collect::<Result<_, _>>()?;
    let judge = profile
        .judge
        .as_ref()
        .map(|raw| {
            raw.parse::<MemberSpec>().map_err(|_| ConfigError::Invalid {
                message: format!("profile `{name}`: cannot parse judge `{raw}`"),
            })
        })
        .transpose()?;
    let quorum = profile.quorum.unwrap_or(members.len());
    if quorum == 0 || quorum > members.len() {
        return Err(ConfigError::Invalid {
            message: format!(
                "profile `{name}`: quorum {quorum} must be nonzero and no greater than member count {}",
                members.len()
            ),
        });
    }
    for (label, value) in [
        ("deadline_secs", profile.deadline_secs),
        ("request_timeout_secs", profile.request_timeout_secs),
    ] {
        if value == Some(0) {
            return Err(ConfigError::Invalid {
                message: format!("profile `{name}`: {label} must be at least 1"),
            });
        }
    }
    let strategy = profile.strategy.clone().unwrap_or_else(|| "judge".to_string());
    crate::strategy_from_name(&strategy)
        .map_err(|error| ConfigError::Invalid { message: format!("profile `{name}`: {error}") })?;
    Ok(Council {
        name: name.to_string(),
        description: profile.description.clone(),
        members,
        judge,
        strategy,
        quorum,
        deadline_secs: profile.deadline_secs,
        request_timeout_secs: profile.request_timeout_secs,
        budget_usd: profile.budget_usd,
    })
}

/// Migrate legacy profile keys in a parsed TOML document, in memory only.
///
/// The previous schema used `models` (member list), `timeout_seconds`
/// (per-request timeout), and `deadline_seconds` (overall run deadline);
/// the current schema uses `members`, `request_timeout_secs`, and
/// `deadline_secs` respectively.
/// Every migration records a warning — nothing is dropped silently.
fn migrate_legacy_profiles(doc: &mut toml::Value, warnings: &mut Vec<String>) {
    let Some(profiles) = doc.get_mut("profiles").and_then(toml::Value::as_table_mut) else {
        return;
    };
    for (name, profile) in profiles.iter_mut() {
        let Some(table) = profile.as_table_mut() else { continue };

        // `models` -> `members`, migrating legacy member strings.
        if let Some(models) = table.remove("models") {
            if table.contains_key("members") {
                warnings.push(format!(
                    "profile `{name}`: both `models` and `members` are set; \
                     using `members` and ignoring the legacy `models` list"
                ));
            } else if let Some(entries) = models.as_array() {
                let migrated = entries
                    .iter()
                    .map(|entry| match entry.as_str() {
                        Some(spec) => {
                            toml::Value::String(migrate_legacy_member(name, spec, warnings))
                        }
                        None => entry.clone(),
                    })
                    .collect();
                table.insert("members".to_string(), toml::Value::Array(migrated));
                warnings.push(format!(
                    "profile `{name}`: legacy key `models` was renamed to `members`; \
                     rename it in the file to silence this warning"
                ));
            } else {
                // Wrong type: put it back so validation reports it honestly.
                table.insert("models".to_string(), models);
            }
        }

        // A `judge` pin keeps its meaning: it names the judging member.
        if let Some(judge) = table.get("judge").and_then(toml::Value::as_str) {
            let migrated = migrate_legacy_member(name, judge, warnings);
            if migrated != judge {
                table.insert("judge".to_string(), toml::Value::String(migrated));
            }
        }

        // `timeout_seconds` was the per-request timeout.
        if let Some(timeout) = table.remove("timeout_seconds") {
            if table.contains_key("request_timeout_secs") {
                warnings.push(format!(
                    "profile `{name}`: legacy `timeout_seconds` ignored because \
                     `request_timeout_secs` is also set"
                ));
            } else {
                table.insert("request_timeout_secs".to_string(), timeout);
                warnings.push(format!(
                    "profile `{name}`: legacy `timeout_seconds` now maps to \
                     `request_timeout_secs`; rename it in the file to silence this warning"
                ));
            }
        }

        // `deadline_seconds` was the overall run deadline.
        if let Some(deadline) = table.remove("deadline_seconds") {
            if table.contains_key("deadline_secs") {
                warnings.push(format!(
                    "profile `{name}`: legacy `deadline_seconds` ignored because \
                     `deadline_secs` is also set"
                ));
            } else {
                table.insert("deadline_secs".to_string(), deadline);
                warnings.push(format!(
                    "profile `{name}`: legacy `deadline_seconds` now maps to `deadline_secs` \
                     (whole-run deadline); rename it in the file to silence this warning"
                ));
            }
        }
    }
}

/// Migrate one legacy member string. Legacy entries could omit the model
/// pin entirely (`codex@xhigh`), which meant the utility's default model;
/// the `glm` utility alias is handled by the [`crate::adapters::Utility`]
/// parser itself.
fn migrate_legacy_member(profile: &str, spec: &str, warnings: &mut Vec<String>) -> String {
    if spec.starts_with("glm:") {
        warnings.push(format!(
            "profile `{profile}`: utility alias `glm` resolves to `opencode`; \
             update the pin to `opencode:...` to silence this warning"
        ));
    }
    if spec.contains(':') {
        return spec.to_string();
    }
    let migrated = match spec.rsplit_once('@') {
        Some((utility, effort)) if !effort.is_empty() => format!("{utility}:default@{effort}"),
        _ => format!("{spec}:default"),
    };
    warnings.push(format!(
        "profile `{profile}`: legacy member `{spec}` has no model pin; \
         interpreted as `{migrated}`"
    ));
    migrated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_toml() {
        let toml = r#"
default_profile = "local"

[profiles.local]
description = "cheap panel"
strategy = "majority-vote"
quorum = 2
deadline_secs = 120
budget_usd = 1.5
members = ["kimi:kimi-code/k3@high", "codex:default@xhigh"]

[adapters.kimi]
cli_path = "/usr/local/bin/kimi"
"#;
        let config = Config::from_toml_str(toml).unwrap();
        assert_eq!(config.default_profile.as_deref(), Some("local"));
        let council = config.resolve_profile(None).unwrap();
        assert_eq!(council.name, "local");
        assert_eq!(council.description.as_deref(), Some("cheap panel"));
        assert_eq!(council.strategy, "majority-vote");
        assert_eq!(council.quorum, 2);
        assert_eq!(council.deadline_secs, Some(120));
        assert_eq!(council.budget_usd, Some(1.5));
        assert_eq!(council.members.len(), 2);
        assert_eq!(
            config.adapter_config("kimi").unwrap().get("cli_path").unwrap().as_str(),
            Some("/usr/local/bin/kimi")
        );
    }

    #[test]
    fn metadata_defaults_apply() {
        let config = Config::from_toml_str(
            r#"
[profiles.min]
members = ["codex:default@xhigh", "kimi:kimi-code/k3@high"]
"#,
        )
        .unwrap();
        let council = config.resolve_profile(Some("min")).unwrap();
        assert_eq!(council.strategy, "judge");
        assert_eq!(council.quorum, 2);
        assert_eq!(council.description, None);
        assert_eq!(council.deadline_secs, None);
        assert_eq!(council.budget_usd, None);
    }

    #[test]
    fn builtin_deep_is_exact() {
        let config = Config::default();
        let council = config.resolve_profile(Some("deep")).unwrap();
        let rendered: Vec<String> = council.members.iter().map(|m| m.to_string()).collect();
        assert_eq!(
            rendered,
            vec![
                "claude:opus@xhigh",
                "claude:claude-fable-5@xhigh",
                "codex:default@xhigh",
                "opencode:zai-coding-plan/glm-5.2@xhigh",
                "kimi:kimi-code/k3@high",
                "grok:grok-4.5@high",
            ]
        );
        assert_eq!(council.strategy, "judge");
        assert_eq!(council.quorum, 3);
        assert_eq!(council.deadline_secs, Some(600));
    }

    #[test]
    fn user_profile_shadows_builtin() {
        let config = Config::from_toml_str(
            r#"
[profiles.deep]
quorum = 1
members = ["codex:default@xhigh"]
"#,
        )
        .unwrap();
        let council = config.resolve_profile(Some("deep")).unwrap();
        assert_eq!(council.members.len(), 1);
        assert_eq!(council.quorum, 1);
        assert_eq!(config.profile_names().iter().filter(|n| *n == "deep").count(), 1);
    }

    #[test]
    fn rejects_zero_and_oversized_quorum() {
        for quorum in [0, 3] {
            let toml =
                format!("[profiles.bad]\nquorum = {quorum}\nmembers = [\"codex:default@xhigh\"]\n");
            assert!(Config::from_toml_str(&toml).is_err(), "quorum {quorum} must fail");
        }
    }

    #[test]
    fn rejects_invalid_effort() {
        let toml = r#"
[profiles.bad]
members = ["kimi:kimi-code/k3@ludicrous"]
"#;
        assert!(Config::from_toml_str(toml).is_err());
    }

    #[test]
    fn rejects_unknown_strategy_before_any_run() {
        let config = Config::from_toml_str(
            r#"
[profiles.bad]
strategy = "judeg"
members = ["codex:default@xhigh"]
"#,
        );
        let error = config.unwrap_err().to_string();
        assert!(error.contains("Unknown strategy: judeg"), "got: {error}");
    }

    #[test]
    fn discovery_precedence() {
        let dir = env::temp_dir().join(format!("caucus-test-{}", std::process::id()));
        let cwd = dir.join("cwd");
        let xdg = dir.join("xdg");
        let home = dir.join("home");
        for d in [&cwd, &xdg, &home] {
            std::fs::create_dir_all(d).unwrap();
        }
        let body = "[profiles.a]\nquorum = 1\nmembers = [\"codex:default@xhigh\"]\n";
        std::fs::write(cwd.join("caucus.toml"), body).unwrap();
        std::fs::create_dir_all(xdg.join("caucus")).unwrap();
        std::fs::write(xdg.join("caucus").join("config.toml"), body).unwrap();
        std::fs::create_dir_all(home.join(".config").join("caucus")).unwrap();
        std::fs::write(home.join(".config").join("caucus").join("config.toml"), body).unwrap();

        // cwd wins over xdg and home.
        let (path, _) = Config::discover_in(None, &cwd, Some(&xdg), Some(&home)).unwrap().unwrap();
        assert_eq!(path, cwd.join("caucus.toml"));

        // xdg wins over home when cwd has no file.
        std::fs::remove_file(cwd.join("caucus.toml")).unwrap();
        let (path, _) = Config::discover_in(None, &cwd, Some(&xdg), Some(&home)).unwrap().unwrap();
        assert_eq!(path, xdg.join("caucus").join("config.toml"));

        // explicit path wins over everything.
        let explicit = dir.join("explicit.toml");
        std::fs::write(&explicit, body).unwrap();
        let (path, _) =
            Config::discover_in(Some(&explicit), &cwd, Some(&xdg), Some(&home)).unwrap().unwrap();
        assert_eq!(path, explicit);

        // nothing found -> None.
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(Config::discover_in(None, &empty, Some(&empty), Some(&empty)).unwrap().is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_schema_migrates_with_warnings() {
        let toml = r#"
[profiles.deep]
models = [
  "claude:claude-opus-4-6@xhigh",
  "claude:claude-fable-5@xhigh",
  "codex@xhigh",
  "glm:zai-coding-plan/glm-5.2@xhigh",
  "kimi:kimi-code/k3@high",
]
judge = "claude:claude-fable-5@xhigh"
quorum = 4
timeout_seconds = 240
deadline_seconds = 900
"#;
        let config = Config::from_toml_str(toml).unwrap();
        let council = config.resolve_profile(Some("deep")).unwrap();

        // `models` -> `members`; pin-less and aliased entries are migrated.
        let rendered: Vec<String> = council.members.iter().map(|m| m.to_string()).collect();
        assert_eq!(
            rendered,
            vec![
                "claude:claude-opus-4-6@xhigh",
                "claude:claude-fable-5@xhigh",
                "codex:default@xhigh",
                "opencode:zai-coding-plan/glm-5.2@xhigh",
                "kimi:kimi-code/k3@high",
            ]
        );
        // Judge meaning is preserved, not discarded.
        assert_eq!(
            council.judge.as_ref().map(ToString::to_string).as_deref(),
            Some("claude:claude-fable-5@xhigh")
        );
        // Both legacy meanings are preserved independently.
        assert_eq!(council.request_timeout_secs, Some(240));
        assert_eq!(council.deadline_secs, Some(900));
        assert_eq!(council.quorum, 4);
        assert_eq!(council.strategy, "judge");

        let warnings = config.warnings.join("\n");
        assert!(warnings.contains("`models` was renamed to `members`"), "got: {warnings}");
        assert!(warnings.contains("alias `glm` resolves to `opencode`"), "got: {warnings}");
        assert!(warnings.contains("`codex@xhigh` has no model pin"), "got: {warnings}");
        assert!(
            warnings.contains("`timeout_seconds` now maps to `request_timeout_secs`"),
            "got: {warnings}"
        );
        assert!(
            warnings.contains("`deadline_seconds` now maps to `deadline_secs`"),
            "got: {warnings}"
        );
    }

    #[test]
    fn legacy_deadline_seconds_maps_to_whole_run_deadline() {
        let config = Config::from_toml_str(
            r#"
[profiles.solo]
members = ["codex:default@xhigh"]
deadline_seconds = 900
"#,
        )
        .unwrap();
        let council = config.resolve_profile(Some("solo")).unwrap();
        assert_eq!(council.deadline_secs, Some(900));
        assert_eq!(council.request_timeout_secs, None);
        assert_eq!(config.warnings.len(), 1);
        assert!(config.warnings[0].contains("`deadline_seconds` now maps to `deadline_secs`"));
    }

    #[test]
    fn modern_keys_take_precedence_over_legacy_duplicates() {
        let config = Config::from_toml_str(
            r#"
[profiles.mixed]
models = ["kimi:kimi-code/k3@high"]
members = ["codex:default@xhigh"]
timeout_seconds = 60
request_timeout_secs = 30
deadline_secs = 120
"#,
        )
        .unwrap();
        let council = config.resolve_profile(Some("mixed")).unwrap();
        assert_eq!(council.members.len(), 1);
        assert_eq!(council.members[0].utility, crate::adapters::Utility::Codex);
        assert_eq!(council.deadline_secs, Some(120));
        assert_eq!(council.request_timeout_secs, Some(30));
        let warnings = config.warnings.join("\n");
        assert!(warnings.contains("ignoring the legacy `models` list"), "got: {warnings}");
        assert!(warnings.contains("`timeout_seconds` ignored"), "got: {warnings}");
    }

    #[test]
    fn designated_judge_is_validated_like_a_member() {
        let config = Config::from_toml_str(
            r#"
[profiles.j]
members = ["codex:default@xhigh", "kimi:kimi-code/k3@high"]
judge = "claude:opus@max"
"#,
        )
        .unwrap();
        let council = config.resolve_profile(Some("j")).unwrap();
        assert_eq!(
            council.judge.as_ref().map(ToString::to_string).as_deref(),
            Some("claude:opus@max")
        );
        assert!(config.warnings.is_empty());

        let bad = Config::from_toml_str(
            r#"
[profiles.j]
members = ["codex:default@xhigh"]
judge = "kimi:kimi-code/k3@xhigh"
"#,
        );
        assert!(bad.is_err(), "an unsupported judge effort must fail validation");
    }

    #[test]
    fn modern_config_produces_no_warnings() {
        let config = Config::from_toml_str(
            r#"
[profiles.min]
members = ["codex:default@xhigh"]
"#,
        )
        .unwrap();
        assert!(config.warnings.is_empty());
    }
}
