//! Shared council helpers: config loading, typed adapter overrides,
//! `MultiProvider` construction from exact member specs, and the zero-key
//! auto council.
//!
//! The auto council is built only from adapters that are reachable without
//! any API keys: ready CLI adapters (Claude, Codex, and the experimental
//! Kimi/opencode/Grok adapters) at documented default pins, and local servers (Ollama,
//! LM Studio) only when a model is configured or safely discoverable. Remote
//! API adapters are always excluded — auto never requires API keys — and
//! every exclusion is reported with its reason.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use caucus_core::adapters::{self, Discovered, MemberSpec, Utility};
use caucus_core::provider::MultiProvider;
use caucus_core::{AdapterOverrides, Config, Council, ProcessLimits, Transport, provider_for_with};
use serde::Deserialize;

/// Default model pin for Claude in the auto council (matches the built-in
/// `deep` profile's pin).
pub const AUTO_CLAUDE_MODEL: &str = "opus";
/// Default model pin for Codex in the auto council (`default` omits `-m`).
pub const AUTO_CODEX_MODEL: &str = "default";
/// Default model pin for the experimental Kimi adapter.
pub const AUTO_KIMI_MODEL: &str = "kimi-code/k3";
/// Default model pin for the experimental opencode adapter.
pub const AUTO_OPENCODE_MODEL: &str = "zai-coding-plan/glm-5.2";
/// Default model pin for the experimental Grok Build adapter.
pub const AUTO_GROK_MODEL: &str = "grok-4.5";
/// Default effort for auto-council CLI members (supported by all five).
pub const AUTO_EFFORT: &str = "high";

/// Load configuration: explicit `--config` path, then the standard discovery
/// order. Returns `(None, Config::default())` when no file exists. Legacy
/// schema migration warnings are surfaced on stderr.
pub fn load_config(explicit: Option<&Path>) -> anyhow::Result<(Option<PathBuf>, Config)> {
    match Config::discover(explicit)? {
        Some((path, config)) => {
            validate_adapter_configs(&config)?;
            for warning in &config.warnings {
                eprintln!("config warning: {warning}");
            }
            Ok((Some(path), config))
        }
        None => Ok((None, Config::default())),
    }
}

/// Validate every adapter section up front, including adapters that are not
/// members of the selected profile. This keeps `doctor` honest and prevents
/// misspelled execution settings from remaining dormant until a later
/// profile happens to use that adapter.
fn validate_adapter_configs(config: &Config) -> anyhow::Result<()> {
    for name in config.adapters.keys() {
        let utility: Utility = name.parse().map_err(|error| {
            anyhow::anyhow!("invalid adapter section `[adapters.{name}]`: {error}")
        })?;
        if utility.as_str() != name {
            anyhow::bail!(
                "invalid adapter section `[adapters.{name}]`: use the canonical adapter id `[adapters.{}]`",
                utility.as_str()
            );
        }
        adapter_config(config, utility)
            .map_err(|error| anyhow::anyhow!("invalid `[adapters.{name}]` config: {error}"))?;
    }
    Ok(())
}

/// Typed `[adapters.<id>]` overrides. Unknown keys are rejected so a typo
/// cannot silently disable a safety or execution setting.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterConfig {
    /// Explicit path to the adapter binary (skips PATH lookup).
    pub binary_path: Option<PathBuf>,
    /// Wall-clock limit per run; the child is killed on expiry.
    pub timeout_secs: Option<u64>,
    /// Stdout cap; excess is truncated.
    pub max_stdout_bytes: Option<usize>,
    /// Stderr cap; excess is truncated.
    pub max_stderr_bytes: Option<usize>,
    /// Exact model pin used by the auto council for local-server adapters
    /// (ollama, lmstudio) when live discovery finds nothing.
    pub model: Option<String>,
    /// Explicit environment passed only to this adapter's child process.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// Parse the typed overrides for one adapter from the loaded config.
pub fn adapter_config(config: &Config, utility: Utility) -> anyhow::Result<AdapterConfig> {
    match config.adapter_config(utility.as_str()) {
        None => Ok(AdapterConfig::default()),
        Some(value) => Ok(AdapterConfig::deserialize(value.clone())?),
    }
}

/// Map typed config overrides into core [`AdapterOverrides`].
#[cfg(test)]
pub fn adapter_overrides(config: &Config, utility: Utility) -> anyhow::Result<AdapterOverrides> {
    adapter_overrides_with_timeout(config, utility, None)
}

fn adapter_overrides_with_timeout(
    config: &Config,
    utility: Utility,
    request_timeout: Option<Duration>,
) -> anyhow::Result<AdapterOverrides> {
    let typed = adapter_config(config, utility)?;
    let mut limits = ProcessLimits::default();
    let mut touched = false;
    if typed.timeout_secs == Some(0) {
        anyhow::bail!("[adapters.{}].timeout_secs must be at least 1", utility.as_str());
    }
    let configured_timeout = typed.timeout_secs.map(Duration::from_secs);
    if let Some(timeout) = match (configured_timeout, request_timeout) {
        (Some(configured), Some(request)) => Some(configured.min(request)),
        (Some(configured), None) => Some(configured),
        (None, Some(request)) => Some(request),
        (None, None) => None,
    } {
        limits.timeout = timeout;
        touched = true;
    }
    if let Some(max) = typed.max_stdout_bytes {
        limits.max_stdout_bytes = max;
        touched = true;
    }
    if let Some(max) = typed.max_stderr_bytes {
        limits.max_stderr_bytes = max;
        touched = true;
    }
    Ok(AdapterOverrides {
        binary_path: typed.binary_path,
        limits: touched.then_some(limits),
        env: typed.env.into_iter().collect(),
    })
}

/// Build a [`MultiProvider`] from a council's exact member specs. Each
/// provider is registered under the member's exact `utility:model@effort`
/// string, so candidates and per-participant debate lookup line up exactly.
pub fn build_council_provider(council: &Council, config: &Config) -> anyhow::Result<MultiProvider> {
    let mut multi = MultiProvider::new();
    let request_timeout =
        council.request_timeout_secs.or(council.deadline_secs).map(Duration::from_secs);
    for member in &council.members {
        let overrides = adapter_overrides_with_timeout(config, member.utility, request_timeout)?;
        let provider = provider_for_with(member, &overrides)?;
        multi = multi.add_shared(member.to_string(), std::sync::Arc::from(provider));
    }
    Ok(multi)
}

/// The judge for strategies that need one, either borrowed from the council
/// provider set or built dedicated when the designated judge is not a member.
pub enum JudgeProvider<'a> {
    Borrowed(&'a dyn caucus_core::LlmProvider),
    Owned(Box<dyn caucus_core::LlmProvider>),
}

impl JudgeProvider<'_> {
    pub fn get(&self) -> &dyn caucus_core::LlmProvider {
        match self {
            Self::Borrowed(p) => *p,
            Self::Owned(p) => p.as_ref(),
        }
    }
}

/// Pick the judge for a council run: the profile's designated `judge` member
/// when set — reusing the council provider when the judge is also a member,
/// building a dedicated one when it is not — and otherwise the first council
/// member's provider. Returns the exact judge member string and its provider.
pub fn select_judge<'a>(
    council: &Council,
    multi: &'a MultiProvider,
    config: &Config,
) -> anyhow::Result<(String, JudgeProvider<'a>)> {
    if let Some(judge) = &council.judge {
        let name = judge.to_string();
        if let Some(provider) = multi.get(&name) {
            return Ok((name, JudgeProvider::Borrowed(provider)));
        }
        let request_timeout =
            council.request_timeout_secs.or(council.deadline_secs).map(Duration::from_secs);
        let overrides = adapter_overrides_with_timeout(config, judge.utility, request_timeout)?;
        return Ok((name, JudgeProvider::Owned(provider_for_with(judge, &overrides)?)));
    }
    let first = multi.models().into_iter().next().expect("council is non-empty");
    Ok((first.to_string(), JudgeProvider::Borrowed(multi.get(first).expect("registered above"))))
}

/// An adapter left out of the auto council, with the honest reason why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoExclusion {
    pub adapter: String,
    pub reason: String,
}

/// The auto council plus the adapters that were excluded from it.
#[derive(Debug, Clone)]
pub struct AutoSelection {
    pub council: Council,
    pub exclusions: Vec<AutoExclusion>,
}

/// Build the zero-key auto council from live discovery. Probes PATH and
/// local-server ports (and, for up local servers, their model lists); never
/// reads credentials.
pub async fn auto_council(config: &Config) -> AutoSelection {
    let discovered = adapters::discover().await;
    let mut configured_models = BTreeMap::new();
    let mut probed_models = BTreeMap::new();
    for d in &discovered {
        let id = d.descriptor.id;
        if let Ok(utility) = id.parse::<Utility>()
            && let Ok(typed) = adapter_config(config, utility)
            && let Some(model) = typed.model
        {
            configured_models.insert(id.to_string(), model);
        }
        if d.descriptor.transport == Transport::LocalServer
            && d.readiness.is_ready()
            && let Some(model) = probe_first_local_model(d.descriptor.base_url.unwrap_or("")).await
        {
            probed_models.insert(id.to_string(), model);
        }
    }
    let (members, exclusions) = select_auto(&discovered, &configured_models, &probed_models);
    let quorum = members.len().clamp(1, 3);
    let council = Council {
        name: "auto".to_string(),
        description: Some(
            "Zero-key council from locally discovered adapters (no API keys required)".to_string(),
        ),
        members,
        judge: None,
        strategy: "judge".to_string(),
        quorum,
        deadline_secs: Some(600),
        request_timeout_secs: None,
        budget_usd: None,
    };
    AutoSelection { council, exclusions }
}

/// Pure selection step for the auto council, split out for testing. Uses
/// documented default pins for CLI adapters; local servers need a configured
/// or probed model; API adapters and ACP are always excluded.
pub fn select_auto(
    discovered: &[Discovered],
    configured_models: &BTreeMap<String, String>,
    probed_models: &BTreeMap<String, String>,
) -> (Vec<MemberSpec>, Vec<AutoExclusion>) {
    let mut members = Vec::new();
    let mut exclusions = Vec::new();
    for d in discovered {
        let id = d.descriptor.id;
        let default_pin = match id {
            "claude" => Some(AUTO_CLAUDE_MODEL),
            "codex" => Some(AUTO_CODEX_MODEL),
            "kimi" => Some(AUTO_KIMI_MODEL),
            "opencode" => Some(AUTO_OPENCODE_MODEL),
            "grok" => Some(AUTO_GROK_MODEL),
            _ => None,
        };
        match d.descriptor.transport {
            Transport::Command => {
                if !d.readiness.is_ready() {
                    exclusions.push(AutoExclusion {
                        adapter: id.to_string(),
                        reason: d.readiness.to_string(),
                    });
                    continue;
                }
                let pin = default_pin.expect("command adapters have a default pin");
                let spec = format!("{id}:{pin}@{AUTO_EFFORT}");
                match spec.parse::<MemberSpec>() {
                    Ok(member) => members.push(member),
                    Err(e) => exclusions.push(AutoExclusion {
                        adapter: id.to_string(),
                        reason: format!("default pin `{spec}` is invalid: {e}"),
                    }),
                }
            }
            Transport::LocalServer => {
                if !d.readiness.is_ready() {
                    exclusions.push(AutoExclusion {
                        adapter: id.to_string(),
                        reason: d.readiness.to_string(),
                    });
                    continue;
                }
                let model = configured_models.get(id).or_else(|| probed_models.get(id));
                match model {
                    Some(model) => match format!("{id}:{model}").parse::<MemberSpec>() {
                        Ok(member) => members.push(member),
                        Err(e) => exclusions.push(AutoExclusion {
                            adapter: id.to_string(),
                            reason: format!("model pin `{model}` is invalid: {e}"),
                        }),
                    },
                    None => exclusions.push(AutoExclusion {
                        adapter: id.to_string(),
                        reason: format!(
                            "server is up but no model could be discovered; pin one with \
                             `[adapters.{id}] model = \"...\"`"
                        ),
                    }),
                }
            }
            Transport::Api => exclusions.push(AutoExclusion {
                adapter: id.to_string(),
                reason: "auto never requires API keys; add it to a profile to use it".to_string(),
            }),
            Transport::Acp => exclusions
                .push(AutoExclusion { adapter: id.to_string(), reason: d.readiness.to_string() }),
        }
    }
    (members, exclusions)
}

/// Best-effort model discovery for a local OpenAI-compatible server, with a
/// short timeout. Ollama: `GET /api/tags` → `models[].name`; LM Studio:
/// `GET /v1/models` → `data[].id`. Returns `None` on any failure — a failed
/// probe simply means the adapter is excluded from auto.
async fn probe_first_local_model(base_url: &str) -> Option<String> {
    let (url, list_path) = local_model_probe(base_url);
    let body = tokio::time::timeout(Duration::from_millis(800), http_get(&url)).await.ok()??;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let entries = json.get(list_path)?.as_array()?;
    for entry in entries {
        let name = entry.get("name").or_else(|| entry.get("id")).and_then(|v| v.as_str());
        if let Some(name) = name
            && !name.trim().is_empty()
            && !name.chars().any(char::is_whitespace)
        {
            return Some(name.to_string());
        }
    }
    None
}

fn local_model_probe(base_url: &str) -> (String, &'static str) {
    let (url, list_path) = if base_url.contains(":11434") {
        // Ollama's `/api/tags` lives at the server root, not under `/v1`.
        let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
        (format!("{root}/api/tags"), "models")
    } else {
        // LM Studio exposes the OpenAI-compatible list at `/v1/models`.
        (format!("{}/models", base_url.trim_end_matches('/')), "data")
    };
    (url, list_path)
}

/// Minimal HTTP/1.1 GET over a raw TCP stream (no TLS, local servers only).
/// Never a shell; only the well-known local probe URLs above reach this.
async fn http_get(url: &str) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let rest = url.split("://").nth(1)?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let mut stream = tokio::net::TcpStream::connect(authority).await.ok()?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).await.ok()?;
    let mut buf = Vec::with_capacity(64 * 1024);
    stream.take(1024 * 1024).read_to_end(&mut buf).await.ok()?;
    let text = String::from_utf8_lossy(&buf);
    let (_, body) = text.split_once("\r\n\r\n")?;
    // Tolerate chunked framing by slicing out the JSON document.
    let start = body.find('{')?;
    let end = body.rfind('}')?;
    (start <= end).then(|| body[start..=end].to_string())
}

#[cfg(test)]
mod tests {
    use caucus_core::adapters::{Readiness, descriptor, descriptors};

    use super::*;

    fn discovered_with(ready_ids: &[&str]) -> Vec<Discovered> {
        descriptors()
            .iter()
            .map(|d| {
                let readiness = match d.transport {
                    Transport::Command => {
                        if ready_ids.contains(&d.id) {
                            Readiness::Ready
                        } else {
                            Readiness::MissingBinary(d.binary.unwrap())
                        }
                    }
                    Transport::LocalServer => {
                        if ready_ids.contains(&d.id) {
                            Readiness::Ready
                        } else {
                            Readiness::ServerDown(d.base_url.unwrap())
                        }
                    }
                    Transport::Api => Readiness::Ready,
                    Transport::Acp => Readiness::Unsupported("not implemented".to_string()),
                };
                Discovered { descriptor: d, readiness }
            })
            .collect()
    }

    #[test]
    fn adapter_overrides_map_typed_fields() {
        let config = Config::from_toml_str(
            r#"
[adapters.claude]
binary_path = "/usr/local/bin/claude"
timeout_secs = 900
max_stdout_bytes = 2097152
max_stderr_bytes = 4096
"#,
        )
        .unwrap();
        let overrides = adapter_overrides(&config, Utility::Claude).unwrap();
        assert_eq!(overrides.binary_path, Some(PathBuf::from("/usr/local/bin/claude")));
        let limits = overrides.limits.expect("limits set");
        assert_eq!(limits.timeout, Duration::from_secs(900));
        assert_eq!(limits.max_stdout_bytes, 2097152);
        assert_eq!(limits.max_stderr_bytes, 4096);
    }

    #[test]
    fn adapter_overrides_default_when_unset_or_partial() {
        let config = Config::from_toml_str(
            r#"
[adapters.kimi]
timeout_secs = 300
env = { NO_COLOR = "1" }
"#,
        )
        .unwrap();
        let kimi = adapter_overrides(&config, Utility::Kimi).unwrap();
        assert_eq!(kimi.binary_path, None);
        assert_eq!(kimi.limits.unwrap().timeout, Duration::from_secs(300));
        assert_eq!(kimi.env, vec![("NO_COLOR".to_string(), "1".to_string())]);
        let codex = adapter_overrides(&config, Utility::Codex).unwrap();
        assert!(codex.binary_path.is_none());
        assert!(codex.limits.is_none());
    }

    #[test]
    fn all_adapter_sections_are_validated_even_when_unused() {
        let typo = Config::from_toml_str(
            r#"
[adapters.kimi]
cli_path = "/usr/local/bin/kimi"
"#,
        )
        .unwrap();
        let error = validate_adapter_configs(&typo).unwrap_err().to_string();
        assert!(error.contains("unknown field `cli_path`"), "got: {error}");

        let unknown = Config::from_toml_str(
            r#"
[adapters.future]
timeout_secs = 30
"#,
        )
        .unwrap();
        let error = validate_adapter_configs(&unknown).unwrap_err().to_string();
        assert!(error.contains("[adapters.future]"), "got: {error}");
    }

    #[test]
    fn profile_request_timeout_reaches_process_limits_and_respects_stricter_adapter_cap() {
        let config = Config::from_toml_str(
            r#"
[adapters.claude]
timeout_secs = 90
"#,
        )
        .unwrap();
        let strict = adapter_overrides_with_timeout(
            &config,
            Utility::Claude,
            Some(Duration::from_secs(240)),
        )
        .unwrap();
        assert_eq!(strict.limits.unwrap().timeout, Duration::from_secs(90));

        let inherited = adapter_overrides_with_timeout(
            &Config::default(),
            Utility::Claude,
            Some(Duration::from_secs(240)),
        )
        .unwrap();
        assert_eq!(inherited.limits.unwrap().timeout, Duration::from_secs(240));
    }

    #[test]
    fn adapter_zero_timeout_is_rejected() {
        let config = Config::from_toml_str(
            r#"
[adapters.claude]
timeout_secs = 0
"#,
        )
        .unwrap();
        let error = adapter_overrides(&config, Utility::Claude).unwrap_err().to_string();
        assert!(error.contains("must be at least 1"), "got: {error}");
    }

    #[test]
    fn auto_includes_ready_clis_at_documented_pins() {
        let discovered = discovered_with(&["claude", "codex", "kimi", "opencode", "grok"]);
        let (members, exclusions) = select_auto(&discovered, &BTreeMap::new(), &BTreeMap::new());
        let rendered: Vec<String> = members.iter().map(|m| m.to_string()).collect();
        assert_eq!(
            rendered,
            vec![
                "claude:opus@high",
                "codex:default@high",
                "kimi:kimi-code/k3@high",
                "opencode:zai-coding-plan/glm-5.2@high",
                "grok:grok-4.5@high",
            ]
        );
        // Everything else is excluded with a reason — nothing silently dropped.
        let excluded: Vec<&str> = exclusions.iter().map(|e| e.adapter.as_str()).collect();
        assert_eq!(excluded, vec!["ollama", "lmstudio", "gemini", "acp"]);
        assert!(exclusions.iter().all(|e| !e.reason.is_empty()));
    }

    #[test]
    fn auto_excludes_missing_binaries_honestly() {
        let discovered = discovered_with(&["codex"]);
        let (members, exclusions) = select_auto(&discovered, &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(members.len(), 1);
        let claude = exclusions.iter().find(|e| e.adapter == "claude").unwrap();
        assert!(claude.reason.contains("missing binary"));
    }

    #[test]
    fn auto_local_servers_need_a_model() {
        // Server up, nothing configured or probed → excluded with guidance.
        let discovered = discovered_with(&["ollama"]);
        let (members, exclusions) = select_auto(&discovered, &BTreeMap::new(), &BTreeMap::new());
        assert!(members.is_empty());
        let ollama = exclusions.iter().find(|e| e.adapter == "ollama").unwrap();
        assert!(ollama.reason.contains("[adapters.ollama]"));

        // Configured model wins; probed model is the fallback.
        let configured = BTreeMap::from([("ollama".to_string(), "llama3.2:latest".to_string())]);
        let (members, _) = select_auto(&discovered, &configured, &BTreeMap::new());
        assert_eq!(members[0].to_string(), "ollama:llama3.2:latest");

        let probed = BTreeMap::from([("ollama".to_string(), "qwen3:8b".to_string())]);
        let (members, _) = select_auto(&discovered, &BTreeMap::new(), &probed);
        assert_eq!(members[0].to_string(), "ollama:qwen3:8b");
    }

    #[test]
    fn local_model_probe_uses_each_servers_actual_listing_endpoint() {
        assert_eq!(
            local_model_probe("http://localhost:11434/v1"),
            ("http://localhost:11434/api/tags".to_string(), "models")
        );
        assert_eq!(
            local_model_probe("http://localhost:1234/v1"),
            ("http://localhost:1234/v1/models".to_string(), "data")
        );
    }

    #[tokio::test]
    async fn local_model_probe_reads_a_real_async_http_response() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let count = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..count]).starts_with("GET /v1/models "));
            let body = r#"{"data":[{"id":"qwen-test"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let model = probe_first_local_model(&format!("http://{address}/v1")).await;
        assert_eq!(model.as_deref(), Some("qwen-test"));
        server.await.unwrap();
    }

    #[test]
    fn auto_never_includes_api_adapters() {
        let discovered = discovered_with(&[]);
        let (members, exclusions) = select_auto(&discovered, &BTreeMap::new(), &BTreeMap::new());
        assert!(members.iter().all(|m| m.utility.descriptor().transport != Transport::Api));
        let gemini = exclusions.iter().find(|e| e.adapter == "gemini").unwrap();
        assert!(gemini.reason.contains("API keys"), "gemini: {}", gemini.reason);
    }

    #[test]
    fn council_provider_uses_exact_member_strings() {
        let council = Council {
            name: "t".to_string(),
            description: None,
            members: vec!["ollama:llama3.2:latest".parse().unwrap()],
            judge: None,
            strategy: "judge".to_string(),
            quorum: 1,
            deadline_secs: None,
            request_timeout_secs: None,
            budget_usd: None,
        };
        let multi = build_council_provider(&council, &Config::default()).unwrap();
        assert_eq!(multi.models(), vec!["ollama:llama3.2:latest"]);
        assert!(multi.get("ollama:llama3.2:latest").is_some());
    }

    #[test]
    fn council_provider_requires_keys_for_api_members() {
        // Avoid mutating the process environment while parallel tests may be
        // reading it. A configured developer key makes this branch inapplicable.
        if std::env::var_os("GEMINI_API_KEY").is_some() {
            return;
        }
        let council = Council {
            name: "t".to_string(),
            description: None,
            members: vec!["gemini:gemini-3.1-pro-preview".parse().unwrap()],
            judge: None,
            strategy: "judge".to_string(),
            quorum: 1,
            deadline_secs: None,
            request_timeout_secs: None,
            budget_usd: None,
        };
        let result = build_council_provider(&council, &Config::default());
        let err = match result {
            Ok(_) => panic!("api member without a key must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("GEMINI_API_KEY"));
    }

    #[test]
    fn select_judge_prefers_designated_judge() {
        let member = |s: &str| s.parse().unwrap();
        let council = Council {
            name: "t".to_string(),
            description: None,
            members: vec![member("ollama:llama3.2:latest"), member("ollama:qwen3:8b")],
            judge: Some(member("ollama:qwen3:8b")),
            strategy: "judge".to_string(),
            quorum: 1,
            deadline_secs: None,
            request_timeout_secs: None,
            budget_usd: None,
        };
        let multi = build_council_provider(&council, &Config::default()).unwrap();

        // Designated judge that is also a member reuses the council provider.
        let (name, judge) = select_judge(&council, &multi, &Config::default()).unwrap();
        assert_eq!(name, "ollama:qwen3:8b");
        assert!(matches!(judge, JudgeProvider::Borrowed(_)));
        assert_eq!(judge.get().transport(), Transport::LocalServer);

        // No designated judge falls back to the first member.
        let council = Council { judge: None, ..council };
        let (name, _) = select_judge(&council, &multi, &Config::default()).unwrap();
        assert_eq!(name, "ollama:llama3.2:latest");

        // A designated judge outside the member list gets a dedicated provider.
        let council = Council { judge: Some(member("ollama:phi4")), ..council };
        let (name, judge) = select_judge(&council, &multi, &Config::default()).unwrap();
        assert_eq!(name, "ollama:phi4");
        assert!(matches!(judge, JudgeProvider::Owned(_)));
    }

    #[test]
    fn descriptor_lookup_for_every_utility() {
        for id in caucus_core::utility_ids() {
            assert!(descriptor(id).unwrap().id == id);
        }
    }
}
