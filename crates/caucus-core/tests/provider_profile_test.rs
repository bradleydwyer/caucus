//! Integration tests for the provider/profile vertical slice:
//! process execution, adapters, discovery, config, and profiles.

use std::time::Duration;

use caucus_core::adapters::{
    self, AdapterOverrides, CommandProvider, CommandSpec, Effort, MemberSpec, PromptDelivery,
    Readiness, Stability, Utility, build_invocation, build_invocation_with,
};
use caucus_core::config::Config;
use caucus_core::error::{ErrorKind, ProviderError};
use caucus_core::process::{ProcessLimits, ProcessSpec, run_argv};
use caucus_core::types::{LlmProvider, Transport};

fn member(spec: &str) -> MemberSpec {
    MemberSpec::parse(spec).unwrap()
}

#[test]
fn deep_council_members_and_metadata_are_exact() {
    let council = Config::default().resolve_profile(Some("deep")).unwrap();
    assert_eq!(council.name, "deep");

    let rendered: Vec<String> = council.members.iter().map(|m| m.to_string()).collect();
    assert_eq!(
        rendered,
        vec![
            "claude:opus@xhigh",
            "claude:claude-fable-5@xhigh",
            "codex:default@xhigh",
            "opencode:zai-coding-plan/glm-5.2@xhigh",
            "kimi:kimi-code/k3@high",
        ]
    );

    assert_eq!(council.strategy, "judge");
    assert_eq!(council.quorum, 3);
    assert_eq!(council.deadline_secs, Some(600));
    assert_eq!(council.description.as_deref(), Some("Broad frontier panel judged by a chair"));
    assert_eq!(council.budget_usd, None);
}

#[test]
fn member_spec_parse_is_exact() {
    let m = member("opencode:zai-coding-plan/glm-5.2@xhigh");
    assert_eq!(m.utility, Utility::Opencode);
    assert_eq!(m.model, "zai-coding-plan/glm-5.2");
    assert_eq!(m.effort, Some(Effort::Xhigh));
    assert_eq!(m.to_string(), "opencode:zai-coding-plan/glm-5.2@xhigh");

    // Effort is optional; the model pin is verbatim, never normalized.
    let m = member("codex:Exact-Model.Name_123");
    assert_eq!(m.utility, Utility::Codex);
    assert_eq!(m.model, "Exact-Model.Name_123");
    assert_eq!(m.effort, None);
    assert_eq!(m.to_string(), "codex:Exact-Model.Name_123");

    // Malformed specs are rejected.
    assert!(MemberSpec::parse("opus@high").is_err()); // missing `:`
    assert!(MemberSpec::parse("bogus:model@high").is_err()); // unknown utility
    assert!(MemberSpec::parse("claude:@high").is_err()); // empty model pin
    assert!(MemberSpec::parse("claude:mo del").is_err()); // whitespace in pin
}

#[test]
fn member_spec_serde_preserves_exact_model_and_validates() {
    let json = r#"{"utility":"claude","model":"claude-opus-4-1-20250805","effort":"xhigh"}"#;
    let m: MemberSpec = serde_json::from_str(json).unwrap();
    assert_eq!(m.utility, Utility::Claude);
    assert_eq!(m.model, "claude-opus-4-1-20250805");
    assert_eq!(m.effort, Some(Effort::Xhigh));

    let round_tripped: MemberSpec =
        serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(round_tripped, m);

    // Serde runs the same validation as parse: unsupported effort fails.
    let bad = r#"{"utility":"kimi","model":"k3","effort":"medium"}"#;
    assert!(serde_json::from_str::<MemberSpec>(bad).is_err());
}

#[test]
fn member_spec_rejects_invalid_effort() {
    // Unknown effort value.
    assert!(MemberSpec::parse("claude:opus@ludicrous").is_err());

    // Known value the utility does not support natively; the error names
    // what is supported.
    let err = MemberSpec::parse("kimi:kimi-code/k3@xhigh").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("xhigh"), "got: {msg}");
    assert!(msg.contains("low, high, max"), "got: {msg}");

    let err = MemberSpec::parse("codex:gpt-5@max").unwrap_err();
    assert!(err.to_string().contains("minimal, low, medium, high, xhigh"));

    // Effort on an adapter with no effort support at all.
    let err = MemberSpec::parse("ollama:llama3.2@high").unwrap_err();
    assert!(err.to_string().contains("does not support effort"));
}

#[test]
fn build_invocation_claude_argv_and_delivery() {
    let inv = build_invocation(&member("claude:opus@xhigh")).unwrap();
    assert_eq!(inv.argv(), ["claude", "-p", "--model", "opus", "--effort", "xhigh"]);
    assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);

    // Without effort the flag is omitted entirely.
    let inv = build_invocation(&member("claude:opus")).unwrap();
    assert_eq!(inv.argv(), ["claude", "-p", "--model", "opus"]);
}

#[test]
fn build_invocation_codex_config_override_and_default_omission() {
    // Effort is delivered as a `-c model_reasoning_effort="..."` config
    // override; the prompt goes to stdin.
    let inv = build_invocation(&member("codex:gpt-5.6-sol@xhigh")).unwrap();
    assert_eq!(
        inv.argv(),
        [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-m",
            "gpt-5.6-sol",
            "-c",
            "model_reasoning_effort=\"xhigh\""
        ]
    );
    assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Stdin);

    // The `default` model omits `-m` entirely.
    let inv = build_invocation(&member("codex:default@minimal")).unwrap();
    assert_eq!(
        inv.argv(),
        ["codex", "exec", "--skip-git-repo-check", "-c", "model_reasoning_effort=\"minimal\""]
    );
    let inv = build_invocation(&member("codex:default")).unwrap();
    assert_eq!(inv.argv(), ["codex", "exec", "--skip-git-repo-check"]);
}

#[test]
fn build_invocation_opencode_variant() {
    let inv = build_invocation(&member("opencode:zai-coding-plan/glm-5.2@xhigh")).unwrap();
    assert_eq!(
        inv.argv(),
        ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "--variant", "xhigh"]
    );
    assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);

    let inv = build_invocation(&member("opencode:glm-5")).unwrap();
    assert_eq!(inv.argv(), ["opencode", "run", "-m", "glm-5"]);
}

#[test]
fn build_invocation_grok_subscription_cli() {
    let inv = build_invocation(&member("grok:grok-4.5@high")).unwrap();
    assert_eq!(
        inv.argv(),
        [
            "grok",
            "--model",
            "grok-4.5",
            "--yolo",
            "--sandbox",
            "read-only",
            "--no-subagents",
            "--output-format",
            "plain",
            "--reasoning-effort",
            "high",
            "-p",
        ]
    );
    assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);
}

#[test]
fn build_invocation_kimi_effort_uses_env_override() {
    let inv = build_invocation(&member("kimi:kimi-code/k3@high")).unwrap();
    let argv = inv.argv();

    // The installed Kimi CLI 0.27.0 has no `--config-file` flag (it exits
    // with "unknown option"), and effort never appears on argv.
    assert!(!argv.iter().any(|a| a == "--config-file"));
    assert!(!argv.iter().any(|a| a == "high"));
    assert_eq!(argv, ["kimi", "-m", "kimi-code/k3", "-p"]);
    assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);

    // Effort rides the documented, non-secret KIMI_MODEL_THINKING_EFFORT
    // override, applied to the child process only.
    assert_eq!(inv.spec.env, vec![(caucus_core::KIMI_EFFORT_ENV.to_string(), "high".to_string())],);
}

#[test]
fn build_invocation_kimi_without_effort_has_no_env() {
    let inv = build_invocation(&member("kimi:kimi-code/k3")).unwrap();
    assert_eq!(inv.argv(), ["kimi", "-m", "kimi-code/k3", "-p"]);
    assert!(inv.spec.env.is_empty());
}

#[test]
fn build_invocation_binary_path_override_replaces_program_only() {
    let overrides =
        AdapterOverrides { binary_path: Some("/opt/claude".into()), ..Default::default() };
    let inv = build_invocation_with(&member("claude:opus@max"), &overrides).unwrap();
    assert_eq!(inv.argv(), ["/opt/claude", "-p", "--model", "opus", "--effort", "max"]);
}

#[test]
fn descriptors_cover_transports_and_stability() {
    let descriptors = adapters::descriptors();
    for transport in [Transport::Api, Transport::Command, Transport::LocalServer, Transport::Acp] {
        assert!(
            descriptors.iter().any(|d| d.transport == transport),
            "no adapter descriptor for {transport}"
        );
    }

    let stable: Vec<&str> =
        descriptors.iter().filter(|d| d.stability == Stability::Stable).map(|d| d.id).collect();
    assert_eq!(stable, ["claude", "codex", "ollama", "lmstudio"]);

    let experimental: Vec<&str> = descriptors
        .iter()
        .filter(|d| d.stability == Stability::Experimental)
        .map(|d| d.id)
        .collect();
    assert_eq!(experimental, ["kimi", "opencode", "gemini", "grok", "acp"]);
}

#[tokio::test]
async fn discover_reports_every_adapter_with_honest_readiness() {
    let found = adapters::discover().await;
    assert_eq!(found.len(), adapters::descriptors().len());

    // ACP is recognized but never ready in this build.
    let acp = found.iter().find(|d| d.descriptor.id == "acp").unwrap();
    assert!(matches!(acp.readiness, Readiness::Unsupported(_)));
    assert!(!acp.readiness.is_ready());

    // The remote Gemini API is ready only when its key variable is actually set
    // (the value itself is never inspected).
    let gemini = found.iter().find(|d| d.descriptor.id == "gemini").unwrap();
    match std::env::var_os("GEMINI_API_KEY") {
        Some(v) if !v.is_empty() => assert!(gemini.readiness.is_ready(), "gemini"),
        _ => assert_eq!(gemini.readiness, Readiness::MissingApiKey("GEMINI_API_KEY")),
    }

    // Grok uses the installed subscription CLI, never an API key.
    let grok = found.iter().find(|d| d.descriptor.id == "grok").unwrap();
    assert!(matches!(grok.readiness, Readiness::Ready | Readiness::MissingBinary("grok")));
}

#[test]
fn legacy_deep_profile_fixture_migrates_with_warnings() {
    // The exact legacy schema found in a real ~/.config/caucus/config.toml.
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/legacy_deep_profile.toml"
    ))
    .expect("legacy fixture must exist");
    let config = Config::from_toml_str(&text).unwrap();

    let council = config.resolve_profile(Some("deep")).unwrap();
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
    // Judge meaning survives the migration.
    assert_eq!(
        council.judge.as_ref().map(ToString::to_string).as_deref(),
        Some("claude:claude-fable-5@xhigh")
    );
    assert_eq!(council.quorum, 4);
    assert_eq!(council.request_timeout_secs, Some(240));
    assert_eq!(council.deadline_secs, Some(900));
    assert_eq!(council.strategy, "judge");

    // Every legacy key produced a warning; nothing was dropped silently.
    let warnings = config.warnings.join("\n");
    for expected in
        ["`models` was renamed to `members`", "`timeout_seconds`", "`deadline_seconds`", "`glm`"]
    {
        assert!(warnings.contains(expected), "missing warning about {expected}: {warnings}");
    }
}

#[test]
fn config_example_parses_and_named_profiles_resolve() {
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/config.toml"))
            .expect("config example must exist");
    let config = Config::from_toml_str(&text).unwrap();

    // The example's default profile resolves to the built-in deep council.
    let council = config.resolve_profile(None).unwrap();
    assert_eq!(council.name, "deep");
    assert_eq!(council.members.len(), 5);

    // User-defined named profiles resolve too.
    let frontier = config.resolve_profile(Some("frontier")).unwrap();
    assert_eq!(frontier.members.len(), 3);
    let local = config.resolve_profile(Some("local")).unwrap();
    assert_eq!(local.members.len(), 2);

    assert!(config.resolve_profile(Some("nonexistent")).is_err());

    // Per-adapter overrides are available as raw TOML values.
    let claude = config.adapter_config("claude").unwrap();
    assert_eq!(claude.get("timeout_secs").unwrap().as_integer(), Some(900));
    let kimi = config.adapter_config("kimi").unwrap();
    assert_eq!(kimi.get("env").unwrap().get("NO_COLOR").unwrap().as_str(), Some("1"));
}

#[cfg(unix)]
#[tokio::test]
async fn run_argv_caps_output_and_flags_truncation() {
    let spec = ProcessSpec::new("/bin/cat").stdin("y".repeat(10_000));
    let limits = ProcessLimits { max_stdout_bytes: 256, ..ProcessLimits::default() };
    let out = run_argv(&spec, &limits).await.unwrap();
    assert!(out.truncated);
    assert_eq!(out.stdout.len(), 256);
}

#[cfg(unix)]
#[tokio::test]
async fn run_argv_timeout_kills_child_promptly() {
    let spec = ProcessSpec::new("/bin/sleep").arg("30");
    let limits = ProcessLimits::default().with_timeout(Duration::from_millis(100));
    let started = std::time::Instant::now();
    let err = run_argv(&spec, &limits).await.unwrap_err();
    assert_eq!(ProviderError::classify(&err), ErrorKind::Timeout);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[cfg(unix)]
#[tokio::test]
async fn run_argv_scrubs_environment_and_applies_overrides() {
    assert!(std::env::var_os("CARGO_MANIFEST_DIR").is_some());
    let spec = ProcessSpec::new("/usr/bin/env");
    let out = run_argv(&spec, &ProcessLimits::default()).await.unwrap();
    assert!(
        !out.stdout.lines().any(|l| l.starts_with("CARGO_MANIFEST_DIR=")),
        "non-allowlisted variable leaked into child: {}",
        out.stdout
    );
    // Allowlisted variables (e.g. PATH) survive the scrub.
    assert!(out.stdout.lines().any(|l| l.starts_with("PATH=")), "got: {}", out.stdout);

    // Explicit non-secret overrides always reach the child.
    let spec = ProcessSpec::new("/usr/bin/env").env("CAUCUS_TEST_VAR", "override-works");
    let out = run_argv(&spec, &ProcessLimits::default()).await.unwrap();
    assert!(out.stdout.lines().any(|l| l == "CAUCUS_TEST_VAR=override-works"));
}

#[cfg(unix)]
#[tokio::test]
async fn command_provider_end_to_end() {
    // Arg delivery: the prompt is the final argv element.
    let spec = CommandSpec::new("/bin/echo").args(["-n"]).prompt_delivery(PromptDelivery::Arg);
    let provider = CommandProvider::new(spec);
    assert_eq!(provider.complete("hello", None).await.unwrap(), "hello");
    assert_eq!(provider.transport(), Transport::Command);

    // Stdin delivery: the prompt is piped to the child.
    let spec = CommandSpec::new("/bin/cat").prompt_delivery(PromptDelivery::Stdin);
    let provider = CommandProvider::new(spec);
    assert_eq!(provider.complete("ping", None).await.unwrap(), "ping");

    // Provider limits bound the child process.
    let spec = CommandSpec::new("/bin/cat").prompt_delivery(PromptDelivery::Stdin);
    let limits = ProcessLimits { max_stdout_bytes: 128, ..ProcessLimits::default() };
    let provider = CommandProvider::new(spec).with_limits(limits);
    let out = provider.run(&"z".repeat(10_000)).await.unwrap();
    assert!(out.truncated);
    assert_eq!(out.stdout.len(), 128);
}
