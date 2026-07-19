//! Provider adapters: registry, member specs, discovery, and factories.
//!
//! Each adapter knows how to reach one kind of model host:
//!
//! - **CLI adapters** (Claude, Codex, Kimi, opencode, Grok) run the installed
//!   binary through [`CommandProvider`] — explicit argv executed via
//!   [`run_argv`], never a shell, and never touching the CLI's credential
//!   store.
//! - **Local servers** (Ollama, LM Studio) and the Gemini remote API are
//!   backed by [`crate::provider::HttpProvider`].
//! - **ACP** is a recognized transport with an honest unsupported status.
//!
//! Effort mappings are evidence-based and never silently dropped: an
//! unsupported effort value is a hard error. Claude, Codex, and opencode
//! take effort as exact argv; Kimi effort (low/high/max) is delivered
//! through the documented `KIMI_MODEL_THINKING_EFFORT` environment
//! override — a non-secret value applied to the child process only. The
//! Kimi CLI (verified 0.27.0) has no `--config-file` flag and no effort
//! argv flag; `KIMI_MODEL_THINKING_EFFORT` is the supported
//! per-invocation mechanism and requires no credential access.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::process::{ProcessLimits, ProcessOutput, ProcessSpec, find_on_path, run_argv};
use crate::provider::HttpProvider;
pub use crate::types::Transport;
use crate::types::{LlmProvider, ProviderOptions};

/// Reasoning effort level requested from a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Effort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Effort {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            "max" => Ok(Self::Max),
            other => anyhow::bail!(
                "unknown effort `{other}` (expected one of: minimal, low, medium, high, xhigh, max)"
            ),
        }
    }
}

/// A known provider utility (CLI tool, local server, remote API, or ACP).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Utility {
    Claude,
    Codex,
    Kimi,
    Opencode,
    Ollama,
    Lmstudio,
    Gemini,
    Grok,
    Acp,
}

impl Utility {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Kimi => "kimi",
            Self::Opencode => "opencode",
            Self::Ollama => "ollama",
            Self::Lmstudio => "lmstudio",
            Self::Gemini => "gemini",
            Self::Grok => "grok",
            Self::Acp => "acp",
        }
    }

    /// This utility's adapter descriptor.
    pub fn descriptor(&self) -> &'static AdapterDescriptor {
        descriptor(self.as_str()).expect("every utility has a descriptor")
    }
}

impl std::fmt::Display for Utility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Utility {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "claude" => Ok(Self::Claude),
            "codex" => Ok(Self::Codex),
            "kimi" => Ok(Self::Kimi),
            "opencode" => Ok(Self::Opencode),
            "ollama" => Ok(Self::Ollama),
            "lmstudio" => Ok(Self::Lmstudio),
            "gemini" => Ok(Self::Gemini),
            "grok" => Ok(Self::Grok),
            "acp" => Ok(Self::Acp),
            "glm" => Ok(Self::Opencode), // legacy alias for the GLM-via-opencode provider
            other => {
                anyhow::bail!("unknown utility `{other}` (known: {})", utility_ids().join(", "))
            }
        }
    }
}

/// One council member: `utility:model@effort`.
///
/// The model pin is *exact*: it is passed verbatim to the adapter, never
/// aliased or normalized. The effort suffix is optional and validated
/// against the utility's native effort support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MemberSpecWire", into = "MemberSpecWire")]
pub struct MemberSpec {
    pub utility: Utility,
    pub model: String,
    pub effort: Option<Effort>,
}

/// Serde wire form; validation happens in the `TryFrom` conversion.
#[derive(Serialize, Deserialize)]
struct MemberSpecWire {
    utility: Utility,
    model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effort: Option<Effort>,
}

impl From<MemberSpec> for MemberSpecWire {
    fn from(m: MemberSpec) -> Self {
        Self { utility: m.utility, model: m.model, effort: m.effort }
    }
}

impl TryFrom<MemberSpecWire> for MemberSpec {
    type Error = anyhow::Error;

    fn try_from(wire: MemberSpecWire) -> Result<Self> {
        let member = Self { utility: wire.utility, model: wire.model, effort: wire.effort };
        member.validate()?;
        Ok(member)
    }
}

impl MemberSpec {
    /// Parse `utility:model@effort`. The utility is split at the first `:`;
    /// the effort suffix is everything after the last `@`.
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        let (utility, rest) = spec.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("member spec `{spec}` must be `utility:model@effort` (missing `:`)")
        })?;
        let utility: Utility = utility.parse()?;
        let (model, effort) = match rest.rsplit_once('@') {
            Some((m, e)) if !e.is_empty() => (m, Some(e.parse()?)),
            Some((m, _)) => (m, None), // trailing `@` with empty effort
            None => (rest, None),
        };
        let member = Self { utility, model: model.to_string(), effort };
        member.validate()?;
        Ok(member)
    }

    /// Validate a non-empty exact model pin and native effort support.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            anyhow::bail!("member spec for `{}` must pin an exact, non-empty model", self.utility);
        }
        if self.model.chars().any(char::is_whitespace) {
            anyhow::bail!("member spec for `{}` has whitespace in the model pin", self.utility);
        }
        if let Some(effort) = self.effort {
            validate_effort(self.utility.descriptor(), effort)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for MemberSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.utility, self.model)?;
        if let Some(e) = &self.effort {
            write!(f, "@{e}")?;
        }
        Ok(())
    }
}

impl std::str::FromStr for MemberSpec {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Release stability advertised by an adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Stability {
    /// First-class, tested adapter.
    Stable,
    /// Works, but the integration may change as the upstream CLI evolves.
    Experimental,
}

impl std::fmt::Display for Stability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => f.write_str("stable"),
            Self::Experimental => f.write_str("experimental"),
        }
    }
}

/// Static description of a provider adapter.
#[derive(Debug, Clone)]
pub struct AdapterDescriptor {
    /// Short id used in member specs (`claude`, `codex`, `opencode`, ...).
    pub id: &'static str,
    pub display_name: &'static str,
    pub transport: Transport,
    pub stability: Stability,
    /// Effort values this adapter accepts natively (empty = no effort support).
    pub efforts: &'static [Effort],
    /// Binary probed on `PATH` (command transports).
    pub binary: Option<&'static str>,
    /// Environment variable holding the API key (api transports).
    pub api_key_env: Option<&'static str>,
    /// Default base URL (local-server transports).
    pub base_url: Option<&'static str>,
    pub notes: &'static str,
}

impl AdapterDescriptor {
    pub fn supports_effort(&self, effort: Effort) -> bool {
        self.efforts.contains(&effort)
    }
}

/// Backwards-compatible alias for [`AdapterDescriptor`].
pub type ProviderDescriptor = AdapterDescriptor;

use Effort::*;

/// All known adapters, in display order.
pub fn descriptors() -> &'static [AdapterDescriptor] {
    &[
        AdapterDescriptor {
            id: "claude",
            display_name: "Claude CLI",
            transport: Transport::Command,
            stability: Stability::Stable,
            // `claude --help`: --effort <level>
            efforts: &[Low, Medium, High, Xhigh, Max],
            binary: Some("claude"),
            api_key_env: None,
            base_url: None,
            notes: "Native effort via `--effort` argv flag",
        },
        AdapterDescriptor {
            id: "codex",
            display_name: "Codex CLI",
            transport: Transport::Command,
            stability: Stability::Stable,
            // `codex debug models`: supported_reasoning_levels per model
            efforts: &[Minimal, Low, Medium, High, Xhigh],
            binary: Some("codex"),
            api_key_env: None,
            base_url: None,
            notes: "Native effort via `-c model_reasoning_effort=...`",
        },
        AdapterDescriptor {
            id: "kimi",
            display_name: "Kimi CLI",
            transport: Transport::Command,
            stability: Stability::Experimental,
            // Kimi CLI `[thinking]` config: support_efforts = ["low", "high", "max"]
            efforts: &[Low, High, Max],
            binary: Some("kimi"),
            api_key_env: None,
            base_url: None,
            notes: "Effort via the KIMI_MODEL_THINKING_EFFORT env override (no argv flag exists)",
        },
        AdapterDescriptor {
            id: "opencode",
            display_name: "opencode / GLM",
            transport: Transport::Command,
            stability: Stability::Experimental,
            // `opencode run --help`: --variant (e.g. minimal, high, max)
            efforts: &[Minimal, Low, Medium, High, Xhigh, Max],
            binary: Some("opencode"),
            api_key_env: None,
            base_url: None,
            notes: "Native effort via `--variant` argv flag",
        },
        AdapterDescriptor {
            id: "ollama",
            display_name: "Ollama (local server)",
            transport: Transport::LocalServer,
            stability: Stability::Stable,
            efforts: &[],
            binary: None,
            api_key_env: None,
            base_url: Some("http://localhost:11434/v1"),
            notes: "Local OpenAI-compatible server, no API key",
        },
        AdapterDescriptor {
            id: "lmstudio",
            display_name: "LM Studio (local server)",
            transport: Transport::LocalServer,
            stability: Stability::Stable,
            efforts: &[],
            binary: None,
            api_key_env: None,
            base_url: Some("http://localhost:1234/v1"),
            notes: "Local OpenAI-compatible server, no API key",
        },
        AdapterDescriptor {
            id: "gemini",
            display_name: "Gemini API",
            transport: Transport::Api,
            stability: Stability::Experimental,
            efforts: &[],
            binary: None,
            api_key_env: Some("GEMINI_API_KEY"),
            base_url: None,
            notes: "Remote API via HttpProvider; ready only when GEMINI_API_KEY is set",
        },
        AdapterDescriptor {
            id: "grok",
            display_name: "Grok Build CLI",
            transport: Transport::Command,
            stability: Stability::Experimental,
            // `grok --help` (0.2.103): --reasoning-effort; the CLI rejects
            // xhigh/max and reports the exact accepted set below.
            efforts: &[Low, Medium, High],
            binary: Some("grok"),
            api_key_env: None,
            base_url: None,
            notes: "Subscription CLI; native effort via `--reasoning-effort` (low, medium, high)",
        },
        AdapterDescriptor {
            id: "acp",
            display_name: "Agent Client Protocol",
            transport: Transport::Acp,
            stability: Stability::Experimental,
            efforts: &[],
            binary: None,
            api_key_env: None,
            base_url: None,
            notes: "Recognized but not supported by this build",
        },
    ]
}

/// Look up a descriptor by id.
pub fn descriptor(id: &str) -> Option<&'static AdapterDescriptor> {
    descriptors().iter().find(|d| d.id == id)
}

/// All registered utility ids, in registry order.
pub fn utility_ids() -> Vec<&'static str> {
    descriptors().iter().map(|d| d.id).collect()
}

/// Validate an effort value against a descriptor's native support.
/// Returns a useful error listing the supported values when unsupported.
pub fn validate_effort(descriptor: &AdapterDescriptor, effort: Effort) -> Result<()> {
    if descriptor.efforts.is_empty() {
        anyhow::bail!(
            "utility `{}` does not support effort levels; drop the `@{effort}` suffix",
            descriptor.id
        );
    }
    if !descriptor.supports_effort(effort) {
        let supported: Vec<&str> = descriptor.efforts.iter().map(Effort::as_str).collect();
        anyhow::bail!(
            "effort `{effort}` is not supported by `{}`; supported: {}",
            descriptor.id,
            supported.join(", ")
        );
    }
    Ok(())
}

/// How the prompt reaches a CLI adapter's child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptDelivery {
    /// The prompt is appended as the final argv element.
    Arg,
    /// The prompt is written to the child's stdin.
    Stdin,
}

/// A fully-specified CLI invocation: an explicit argv template plus the
/// prompt delivery mode. Never a shell string.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    /// Executable name or path (looked up on PATH if not absolute).
    pub program: String,
    /// Arguments, in order, each passed verbatim — no shell expansion.
    pub args: Vec<String>,
    /// How the per-completion prompt is delivered.
    pub prompt_delivery: PromptDelivery,
    /// Explicit environment overrides for the child. Values are never logged.
    pub env: Vec<(String, String)>,
    /// Exact adapter-specific parent variables that may be inherited.
    pub inherit_env: Vec<String>,
    /// Insert `--` before an argv-delivered prompt so a leading hyphen cannot
    /// be interpreted as another CLI flag. Used only by CLIs whose prompt is
    /// a positional argument (not an option value).
    end_of_options_before_prompt: bool,
}

impl CommandSpec {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            prompt_delivery: PromptDelivery::Arg,
            env: Vec::new(),
            inherit_env: Vec::new(),
            end_of_options_before_prompt: false,
        }
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn prompt_delivery(mut self, delivery: PromptDelivery) -> Self {
        self.prompt_delivery = delivery;
        self
    }

    /// Add an explicit environment override for the child.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn inherit_env(mut self, key: impl Into<String>) -> Self {
        self.inherit_env.push(key.into());
        self
    }

    pub fn end_of_options_before_prompt(mut self) -> Self {
        self.end_of_options_before_prompt = true;
        self
    }

    /// The full argv vector (program first), without the prompt, for tests
    /// and receipts.
    pub fn argv(&self) -> Vec<String> {
        std::iter::once(self.program.clone()).chain(self.args.iter().cloned()).collect()
    }

    /// Materialize a runnable [`ProcessSpec`] for one prompt.
    fn process_spec(&self, prompt: &str) -> ProcessSpec {
        let mut spec = ProcessSpec::new(self.program.clone()).args(self.args.clone());
        match self.prompt_delivery {
            PromptDelivery::Arg => {
                if self.end_of_options_before_prompt {
                    spec = spec.arg("--");
                }
                spec = spec.arg(prompt);
            }
            PromptDelivery::Stdin => spec = spec.stdin(prompt),
        }
        for (key, value) in &self.env {
            spec = spec.env(key.clone(), value.clone());
        }
        for key in &self.inherit_env {
            spec = spec.inherit_env(key.clone());
        }
        spec
    }
}

/// Per-adapter overrides for discovery and execution. Everything defaults
/// to the descriptor's values; overrides never change argv shape, only
/// where the binary lives and how the process is bounded.
#[derive(Debug, Clone, Default)]
pub struct AdapterOverrides {
    /// Explicit path to the adapter binary (skips PATH lookup).
    pub binary_path: Option<PathBuf>,
    /// Process limits (timeout, output caps) for command adapters.
    pub limits: Option<ProcessLimits>,
    /// Explicit environment values from `[adapters.<id>.env]`.
    pub env: Vec<(String, String)>,
}

/// An [`LlmProvider`] that runs a CLI adapter via [`run_argv`]: explicit
/// argv, scrubbed environment, never a shell.
pub struct CommandProvider {
    spec: CommandSpec,
    limits: ProcessLimits,
    options: ProviderOptions,
}

impl CommandProvider {
    pub fn new(spec: CommandSpec) -> Self {
        Self { spec, limits: ProcessLimits::default(), options: ProviderOptions::default() }
    }

    pub fn with_limits(mut self, limits: ProcessLimits) -> Self {
        self.limits = limits;
        self.options.timeout = limits.timeout;
        self.options.max_output_bytes = limits.max_stdout_bytes;
        self
    }

    pub fn spec(&self) -> &CommandSpec {
        &self.spec
    }

    /// Run one completion and return the raw process output.
    pub async fn run(&self, prompt: &str) -> Result<ProcessOutput> {
        let spec = self.spec.process_spec(prompt);
        run_argv(&spec, &self.limits).await
    }
}

#[async_trait::async_trait]
impl LlmProvider for CommandProvider {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let prompt = match system {
            Some(system) => format!("System instructions:\n{system}\n\nUser request:\n{prompt}"),
            None => prompt.to_string(),
        };
        let out = self.run(&prompt).await?;
        Ok(out.stdout.trim_end().to_string())
    }

    fn transport(&self) -> Transport {
        Transport::Command
    }

    fn options(&self) -> ProviderOptions {
        self.options
    }
}

/// ACP is a recognized transport that this build does not implement. Every
/// completion returns an explicit unsupported error.
pub struct AcpProvider {
    model: String,
}

impl AcpProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self { model: model.into() }
    }
}

#[async_trait::async_trait]
impl LlmProvider for AcpProvider {
    async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
        anyhow::bail!(
            "ACP transport is recognized but not supported by this build (member model `{}`)",
            self.model
        )
    }

    fn transport(&self) -> Transport {
        Transport::Acp
    }
}

/// The environment variable the Kimi CLI (≥ 0.27, verified against the
/// installed 0.27.0 binary and the official docs) reads to force a thinking
/// effort on the wire. Non-secret; applied to the child process only.
pub const KIMI_EFFORT_ENV: &str = "KIMI_MODEL_THINKING_EFFORT";

/// A fully-built command-adapter invocation: the argv template plus any
/// non-secret environment overrides. Effort is already baked into the argv
/// (or the environment, for Kimi); the prompt is supplied per completion by
/// [`CommandProvider`].
pub struct CommandInvocation {
    pub spec: CommandSpec,
}

impl CommandInvocation {
    /// The full argv vector (program first), for tests and receipts.
    pub fn argv(&self) -> Vec<String> {
        self.spec.argv()
    }
}

/// Build the exact invocation for a CLI adapter member. Evidence:
///
/// - claude: `claude -p --model <m> [--effort <e>] <prompt>`
/// - codex: `codex exec [--skip-git-repo-check] [-m <m>]
///   [-c model_reasoning_effort="<e>"]`, prompt on stdin; model `default`
///   omits `-m` entirely
/// - kimi: `kimi -m <m> -p <prompt>`; effort is delivered via the
///   non-secret [`KIMI_EFFORT_ENV`] environment override — the Kimi CLI
///   (verified 0.27.0) has no `--config-file` or effort argv flag, and its
///   authenticated state under `$KIMI_CODE_HOME` is left completely untouched
/// - opencode: `opencode run -m <provider/model> [--variant <e>] <prompt>`
/// - grok: `grok --model <m> [--reasoning-effort <e>] --sandbox read-only
///   --no-subagents --yolo --output-format plain -p <prompt>`
pub fn build_invocation(member: &MemberSpec) -> Result<CommandInvocation> {
    build_invocation_with(member, &AdapterOverrides::default())
}

/// Like [`build_invocation`], with an optional binary path override.
pub fn build_invocation_with(
    member: &MemberSpec,
    overrides: &AdapterOverrides,
) -> Result<CommandInvocation> {
    member.validate()?;
    let descriptor = member.utility.descriptor();
    if descriptor.transport != Transport::Command {
        anyhow::bail!("`{}` is not a command adapter", descriptor.id);
    }
    let program = overrides
        .binary_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| descriptor.binary.expect("command adapter has binary").to_string());
    let model = member.model.as_str();
    let effort = member.effort;

    let mut spec = match member.utility {
        Utility::Claude => {
            let mut spec = CommandSpec::new(program)
                .args(["-p", "--model", model])
                .end_of_options_before_prompt();
            if let Some(e) = effort {
                spec = spec.args(["--effort", e.as_str()]);
            }
            spec
        }
        Utility::Codex => {
            let mut spec = CommandSpec::new(program)
                .args(["exec", "--skip-git-repo-check"])
                .prompt_delivery(PromptDelivery::Stdin);
            if model != "default" {
                spec = spec.args(["-m", model]);
            }
            if let Some(e) = effort {
                spec = spec.args(["-c"]).args([format!("model_reasoning_effort=\"{e}\"")]);
            }
            spec
        }
        Utility::Kimi => {
            let mut spec = CommandSpec::new(program).args(["-m", model]);
            if let Some(e) = effort {
                // Supported per-invocation effort channel: a non-secret env
                // override the Kimi CLI reads at startup. Never argv (no flag
                // exists) and never a config file (`--config-file` does not
                // exist; writing one would not be read).
                spec = spec.env(KIMI_EFFORT_ENV, e.as_str());
            }
            // `-p` takes the prompt as its value; CommandProvider appends it
            // as the final argv element.
            spec.args(["-p"])
        }
        Utility::Opencode => {
            let mut spec =
                CommandSpec::new(program).args(["run", "-m", model]).end_of_options_before_prompt();
            if let Some(e) = effort {
                spec = spec.args(["--variant", e.as_str()]);
            }
            spec
        }
        Utility::Grok => {
            let mut spec = CommandSpec::new(program).args([
                "--model",
                model,
                "--yolo",
                "--sandbox",
                "read-only",
                "--no-subagents",
                "--output-format",
                "plain",
            ]);
            if let Some(e) = effort {
                spec = spec.args(["--reasoning-effort", e.as_str()]);
            }
            // `-p` consumes the following argv element as the prompt. Keep it
            // last so intervening flags cannot become the prompt value.
            spec.args(["-p"])
        }
        other => anyhow::bail!("no command adapter registered for `{other}`"),
    };

    // Inherit only variables that this exact adapter can legitimately use.
    // The subprocess runtime still clears every other ambient variable.
    for key in inherited_env_for(member.utility) {
        spec = spec.inherit_env(key);
    }
    for (key, value) in &overrides.env {
        spec = spec.env(key.clone(), value.clone());
    }

    Ok(CommandInvocation { spec })
}

const COMMON_COMMAND_ENV: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
];

fn inherited_env_for(utility: Utility) -> Vec<&'static str> {
    let mut names = COMMON_COMMAND_ENV.to_vec();
    names.extend(match utility {
        Utility::Claude => &[
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CONFIG_DIR",
        ][..],
        Utility::Codex => &["OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME"][..],
        Utility::Kimi => &["KIMI_CODE_HOME"][..],
        Utility::Opencode => &[
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_DIR",
            "ZAI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ][..],
        Utility::Grok => &["GROK_HOME", "GROK_DISABLE_AUTOUPDATER", "XAI_API_KEY"][..],
        _ => &[][..],
    });
    names
}

/// Build a ready-to-use provider for a validated [`MemberSpec`].
///
/// Command adapters return a [`CommandProvider`]; local servers return an
/// [`HttpProvider::ollama`] / [`HttpProvider::lmstudio`]; remote APIs return
/// an [`HttpProvider`] keyed from the descriptor's `api_key_env`; ACP
/// returns the honestly-unsupported [`AcpProvider`].
pub fn provider_for(member: &MemberSpec) -> Result<Box<dyn LlmProvider>> {
    provider_for_with(member, &AdapterOverrides::default())
}

/// Like [`provider_for`], with per-adapter overrides.
pub fn provider_for_with(
    member: &MemberSpec,
    overrides: &AdapterOverrides,
) -> Result<Box<dyn LlmProvider>> {
    member.validate()?;
    match member.utility.descriptor().transport {
        Transport::Command => {
            let invocation = build_invocation_with(member, overrides)?;
            let mut provider = CommandProvider::new(invocation.spec);
            if let Some(limits) = overrides.limits {
                provider = provider.with_limits(limits);
            }
            Ok(Box::new(provider))
        }
        Transport::LocalServer => {
            let mut provider = match member.utility {
                Utility::Ollama => HttpProvider::ollama(member.model.clone()),
                Utility::Lmstudio => HttpProvider::lmstudio(member.model.clone()),
                other => anyhow::bail!("no local-server adapter registered for `{other}`"),
            };
            if let Some(limits) = overrides.limits {
                provider = provider.with_timeout(limits.timeout);
            }
            Ok(Box::new(provider))
        }
        Transport::Api => {
            let descriptor = member.utility.descriptor();
            let env = descriptor.api_key_env.expect("api adapter has api_key_env");
            let key = std::env::var(env).map_err(|_| {
                anyhow::anyhow!(
                    "utility `{}` requires the `{env}` environment variable",
                    member.utility
                )
            })?;
            let mut provider = match member.utility {
                Utility::Gemini => HttpProvider::gemini(key, member.model.clone()),
                other => anyhow::bail!("no api adapter registered for `{other}`"),
            };
            if let Some(limits) = overrides.limits {
                provider = provider.with_timeout(limits.timeout);
            }
            Ok(Box::new(provider))
        }
        Transport::Acp => Ok(Box::new(AcpProvider::new(member.model.clone()))),
    }
}

/// How reachable an adapter is right now, determined without any credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// Binary found on PATH, local server port accepting connections, or the
    /// API key environment variable is set.
    Ready,
    /// CLI binary not found on PATH.
    MissingBinary(&'static str),
    /// Local server not listening on its well-known port.
    ServerDown(&'static str),
    /// Remote API whose key environment variable is unset or empty.
    MissingApiKey(&'static str),
    /// Recognized but not implemented (e.g. ACP).
    Unsupported(String),
}

impl Readiness {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

impl std::fmt::Display for Readiness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => f.write_str("ready"),
            Self::MissingBinary(bin) => write!(f, "missing binary `{bin}`"),
            Self::ServerDown(addr) => write!(f, "no server at {addr}"),
            Self::MissingApiKey(env) => write!(f, "missing API key `{env}`"),
            Self::Unsupported(why) => write!(f, "unsupported: {why}"),
        }
    }
}

/// A descriptor paired with its current readiness.
#[derive(Debug, Clone)]
pub struct Discovered {
    pub descriptor: &'static AdapterDescriptor,
    pub readiness: Readiness,
}

/// Discover every known adapter without reading any credentials: CLI
/// adapters check PATH, local servers probe their ports, remote APIs check
/// only that their key environment variable is set (never its value — the
/// CLI's dotenv loading runs before discovery, so `.env` keys count), ACP
/// reports its honest unsupported status.
pub async fn discover() -> Vec<Discovered> {
    let mut out = Vec::new();
    for d in descriptors() {
        let readiness = match d.transport {
            Transport::Command => binary_readiness(d.binary.expect("command adapter has binary")),
            Transport::LocalServer => {
                port_readiness(d.base_url.expect("local-server adapter has base url")).await
            }
            Transport::Api => {
                api_key_readiness(d.api_key_env.expect("api adapter has api_key_env"))
            }
            Transport::Acp => Readiness::Unsupported(
                "ACP transport is recognized but not implemented in this build".to_string(),
            ),
        };
        out.push(Discovered { descriptor: d, readiness });
    }
    out
}

/// Remote-API readiness: honest about the key environment variable. Only
/// presence is checked — the value is never read, logged, or probed. An
/// explicit env var or one loaded from a `.env` file (the CLI loads dotenv
/// before any command runs) both count; an unset or empty variable does not.
fn api_key_readiness(api_key_env: &'static str) -> Readiness {
    api_key_readiness_value(api_key_env, std::env::var_os(api_key_env))
}

fn api_key_readiness_value(
    api_key_env: &'static str,
    value: Option<std::ffi::OsString>,
) -> Readiness {
    match value {
        Some(value) if !value.is_empty() => Readiness::Ready,
        _ => Readiness::MissingApiKey(api_key_env),
    }
}

fn binary_readiness(binary: &'static str) -> Readiness {
    if find_on_path(binary).is_some() { Readiness::Ready } else { Readiness::MissingBinary(binary) }
}

async fn port_readiness(base_url: &'static str) -> Readiness {
    let authority = base_url
        .strip_prefix("http://")
        .or_else(|| base_url.strip_prefix("https://"))
        .unwrap_or(base_url);
    let authority = authority.split('/').next().unwrap_or(authority);
    let connect = tokio::net::TcpStream::connect(authority);
    match tokio::time::timeout(Duration::from_millis(300), connect).await {
        Ok(Ok(_)) => Readiness::Ready,
        _ => Readiness::ServerDown(base_url),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(id: &str) -> &'static AdapterDescriptor {
        descriptor(id).unwrap()
    }

    fn member(spec: &str) -> MemberSpec {
        MemberSpec::parse(spec).unwrap()
    }

    #[test]
    fn registry_covers_all_transports_and_utilities() {
        for transport in
            [Transport::Command, Transport::LocalServer, Transport::Api, Transport::Acp]
        {
            assert!(
                descriptors().iter().any(|d| d.transport == transport),
                "no adapter descriptor for transport {transport}"
            );
        }
        let ids = utility_ids();
        for expected in
            ["claude", "codex", "kimi", "opencode", "ollama", "lmstudio", "gemini", "grok", "acp"]
        {
            assert!(ids.contains(&expected), "missing adapter `{expected}`");
        }
    }

    #[test]
    fn descriptor_effort_support_is_exact() {
        assert_eq!(desc("claude").efforts, [Low, Medium, High, Xhigh, Max]);
        assert_eq!(desc("codex").efforts, [Minimal, Low, Medium, High, Xhigh]);
        assert_eq!(desc("kimi").efforts, [Low, High, Max]);
        assert_eq!(desc("opencode").efforts, [Minimal, Low, Medium, High, Xhigh, Max]);
        assert_eq!(desc("grok").efforts, [Low, Medium, High]);
        assert!(desc("ollama").efforts.is_empty());
        assert!(desc("lmstudio").efforts.is_empty());
    }

    #[test]
    fn member_spec_parses_and_round_trips() {
        let m = member("opencode:zai-coding-plan/glm-5.2@xhigh");
        assert_eq!(m.utility, Utility::Opencode);
        assert_eq!(m.model, "zai-coding-plan/glm-5.2");
        assert_eq!(m.effort, Some(Effort::Xhigh));
        assert_eq!(m.to_string(), "opencode:zai-coding-plan/glm-5.2@xhigh");
        assert_eq!(member("ollama:llama3.2:latest").model, "llama3.2:latest");
    }

    #[test]
    fn member_spec_rejects_bad_input() {
        assert!(MemberSpec::parse("opus@high").is_err()); // missing `:`
        assert!(MemberSpec::parse("bogus:model@high").is_err()); // unknown utility
        assert!(MemberSpec::parse("claude:@high").is_err()); // empty model
        assert!(MemberSpec::parse("claude:mo del").is_err()); // whitespace
        assert!(MemberSpec::parse("claude:model@ludicrous").is_err()); // bad effort
    }

    #[test]
    fn member_spec_serde_validates_and_preserves_exact_model() {
        let json = r#"{"utility":"claude","model":"claude-opus-4-1-20250805","effort":"xhigh"}"#;
        let m: MemberSpec = serde_json::from_str(json).unwrap();
        assert_eq!(m.model, "claude-opus-4-1-20250805");
        assert_eq!(m.effort, Some(Effort::Xhigh));
        let bad = r#"{"utility":"kimi","model":"k3","effort":"medium"}"#;
        assert!(serde_json::from_str::<MemberSpec>(bad).is_err());
    }

    #[test]
    fn claude_argv_uses_print_model_and_effort_flags() {
        let inv = build_invocation(&member("claude:opus@xhigh")).unwrap();
        assert_eq!(inv.argv(), ["claude", "-p", "--model", "opus", "--effort", "xhigh"]);
        assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);
        assert!(inv.spec.inherit_env.contains(&"CLAUDE_CONFIG_DIR".to_string()));
        assert!(inv.spec.inherit_env.contains(&"HTTPS_PROXY".to_string()));
    }

    #[test]
    fn claude_argv_without_effort_omits_flag() {
        let inv = build_invocation(&member("claude:opus")).unwrap();
        assert_eq!(inv.argv(), ["claude", "-p", "--model", "opus"]);
    }

    #[test]
    fn codex_argv_uses_config_override_for_effort() {
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
    }

    #[test]
    fn codex_default_model_omits_model_flag() {
        let inv = build_invocation(&member("codex:default@minimal")).unwrap();
        assert_eq!(
            inv.argv(),
            ["codex", "exec", "--skip-git-repo-check", "-c", "model_reasoning_effort=\"minimal\""]
        );
    }

    #[test]
    fn codex_rejects_max_effort() {
        let err = MemberSpec::parse("codex:gpt-5@max").unwrap_err();
        assert!(err.to_string().contains("minimal, low, medium, high, xhigh"));
    }

    #[test]
    fn kimi_effort_uses_env_override_never_argv_or_files() {
        let inv = build_invocation(&member("kimi:kimi-code/k3@high")).unwrap();
        let argv = inv.argv();
        // The effort value itself never appears on argv, and no config-file
        // flag is emitted (the installed Kimi CLI 0.27.0 rejects it).
        assert!(!argv.iter().any(|a| a == "high"));
        assert!(!argv.iter().any(|a| a == "--config-file"));
        assert_eq!(argv, ["kimi", "-m", "kimi-code/k3", "-p"]);
        // Effort is delivered via the documented, non-secret env override.
        assert_eq!(inv.spec.env, vec![(KIMI_EFFORT_ENV.to_string(), "high".to_string())],);
    }

    #[test]
    fn kimi_prompt_is_delivered_as_final_argv_element() {
        let inv = build_invocation(&member("kimi:kimi-code/k3")).unwrap();
        assert_eq!(inv.argv(), ["kimi", "-m", "kimi-code/k3", "-p"]);
        assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);
        assert!(inv.spec.env.is_empty(), "no effort → no env override");
    }

    #[test]
    fn kimi_rejects_unsupported_effort_with_supported_list() {
        let err = MemberSpec::parse("kimi:k3@xhigh").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("xhigh"), "got: {msg}");
        assert!(msg.contains("low, high, max"), "got: {msg}");
    }

    #[test]
    fn opencode_argv_uses_variant_for_effort() {
        let inv = build_invocation(&member("opencode:zai-coding-plan/glm-5.2@xhigh")).unwrap();
        assert_eq!(
            inv.argv(),
            ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "--variant", "xhigh"]
        );
        assert_eq!(inv.spec.prompt_delivery, PromptDelivery::Arg);
    }

    #[test]
    fn grok_argv_uses_subscription_cli_in_read_only_mode() {
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
        assert!(inv.spec.inherit_env.contains(&"GROK_HOME".to_string()));
    }

    #[test]
    fn grok_rejects_effort_above_installed_cli_contract() {
        let error = MemberSpec::parse("grok:grok-4.5@xhigh").unwrap_err().to_string();
        assert!(error.contains("low, medium, high"), "got: {error}");
    }

    #[test]
    fn effort_on_non_effort_adapter_is_rejected() {
        let err = MemberSpec::parse("ollama:llama3.2@high").unwrap_err();
        assert!(err.to_string().contains("does not support effort"));
    }

    #[test]
    fn binary_path_override_replaces_program_only() {
        let overrides =
            AdapterOverrides { binary_path: Some("/opt/claude".into()), ..Default::default() };
        let inv = build_invocation_with(&member("claude:opus@max"), &overrides).unwrap();
        assert_eq!(inv.argv(), ["/opt/claude", "-p", "--model", "opus", "--effort", "max"]);
    }

    #[test]
    fn configured_adapter_environment_reaches_only_that_invocation() {
        let overrides = AdapterOverrides {
            env: vec![("CUSTOM_ENDPOINT".into(), "https://example.invalid".into())],
            ..Default::default()
        };
        let inv = build_invocation_with(&member("claude:opus@high"), &overrides).unwrap();
        assert!(
            inv.spec.env.contains(&("CUSTOM_ENDPOINT".into(), "https://example.invalid".into()))
        );
        let plain = build_invocation(&member("codex:default@high")).unwrap();
        assert!(!plain.spec.env.iter().any(|(key, _)| key == "CUSTOM_ENDPOINT"));
    }

    #[test]
    fn positional_prompts_starting_with_hyphen_are_not_parsed_as_flags() {
        for spec in ["claude:opus@high", "opencode:zai-coding-plan/glm-5.2@high"] {
            let invocation = build_invocation(&member(spec)).unwrap();
            let process = invocation.spec.process_spec("--not-a-real-flag");
            assert_eq!(
                &process.args[process.args.len() - 2..],
                ["--", "--not-a-real-flag"],
                "{spec}"
            );
        }

        let kimi = build_invocation(&member("kimi:kimi-code/k3@high")).unwrap();
        let process = kimi.spec.process_spec("--still-the-prompt");
        assert_eq!(&process.args[process.args.len() - 2..], ["-p", "--still-the-prompt"]);
    }

    #[tokio::test]
    async fn command_provider_runs_explicit_argv_without_shell() {
        let spec = CommandSpec::new("/bin/echo").args(["-n"]).prompt_delivery(PromptDelivery::Arg);
        let provider = CommandProvider::new(spec);
        assert_eq!(provider.complete("hello", None).await.unwrap(), "hello");
        assert_eq!(provider.transport(), Transport::Command);

        let spec = CommandSpec::new("/bin/cat").prompt_delivery(PromptDelivery::Stdin);
        let provider = CommandProvider::new(spec);
        assert_eq!(provider.complete("piped", None).await.unwrap(), "piped");

        let spec = CommandSpec::new("/bin/cat").prompt_delivery(PromptDelivery::Stdin);
        let provider = CommandProvider::new(spec);
        let output = provider.complete("question", Some("be precise")).await.unwrap();
        assert!(output.contains("System instructions:\nbe precise"));
        assert!(output.contains("User request:\nquestion"));
    }

    #[tokio::test]
    async fn acp_provider_is_explicitly_unsupported() {
        let provider = AcpProvider::new("some-agent");
        let err = provider.complete("hi", None).await.unwrap_err();
        assert!(err.to_string().contains("not supported"));
        assert_eq!(provider.transport(), Transport::Acp);
    }

    #[tokio::test]
    async fn acp_reports_honest_unsupported_status() {
        let found = discover().await;
        let acp = found.iter().find(|f| f.descriptor.id == "acp").unwrap();
        assert!(matches!(acp.readiness, Readiness::Unsupported(_)));
        assert!(!acp.readiness.is_ready());
    }

    #[test]
    fn glm_alias_resolves_to_opencode() {
        // Legacy member specs used `glm` for the GLM-via-opencode provider.
        let m = member("glm:zai-coding-plan/glm-5.2@xhigh");
        assert_eq!(m.utility, Utility::Opencode);
        // Canonical rendering always uses the real utility id.
        assert_eq!(m.to_string(), "opencode:zai-coding-plan/glm-5.2@xhigh");
    }

    #[test]
    fn api_readiness_is_honest_about_key_presence() {
        assert_eq!(
            api_key_readiness_value("CAUCUS_TEST_API_KEY", Some("k".into())),
            Readiness::Ready
        );
        assert_eq!(
            api_key_readiness_value("CAUCUS_TEST_API_KEY", Some("".into())),
            Readiness::MissingApiKey("CAUCUS_TEST_API_KEY")
        );
        assert_eq!(
            api_key_readiness_value("CAUCUS_TEST_API_KEY", None),
            Readiness::MissingApiKey("CAUCUS_TEST_API_KEY")
        );
        assert!(!Readiness::MissingApiKey("X").is_ready());
    }

    #[tokio::test]
    async fn api_adapter_readiness_matches_key_presence() {
        let found = discover().await;
        let d = found.iter().find(|d| d.descriptor.id == "gemini").unwrap();
        match std::env::var_os("GEMINI_API_KEY") {
            Some(v) if !v.is_empty() => assert!(d.readiness.is_ready(), "gemini"),
            _ => assert_eq!(d.readiness, Readiness::MissingApiKey("GEMINI_API_KEY")),
        }

        let grok = found.iter().find(|d| d.descriptor.id == "grok").unwrap();
        assert!(matches!(grok.readiness, Readiness::Ready | Readiness::MissingBinary("grok")));
    }

    #[tokio::test]
    async fn discovery_includes_every_descriptor() {
        let found = discover().await;
        assert_eq!(found.len(), descriptors().len());
    }

    #[test]
    fn provider_factory_dispatches_by_transport() {
        let cli = provider_for(&member("kimi:kimi-code/k3@high")).unwrap();
        assert_eq!(cli.transport(), Transport::Command);

        let grok = provider_for(&member("grok:grok-4.5@high")).unwrap();
        assert_eq!(grok.transport(), Transport::Command);

        let local = provider_for(&member("ollama:llama3.2")).unwrap();
        assert_eq!(local.transport(), Transport::LocalServer);

        let studio = provider_for(&member("lmstudio:qwen2.5-coder")).unwrap();
        assert_eq!(studio.transport(), Transport::LocalServer);

        let acp = provider_for(&member("acp:some-agent")).unwrap();
        assert_eq!(acp.transport(), Transport::Acp);
    }

    #[test]
    fn http_provider_honors_adapter_timeout_override() {
        let overrides = AdapterOverrides {
            limits: Some(ProcessLimits::default().with_timeout(Duration::from_secs(321))),
            ..Default::default()
        };
        let provider = provider_for_with(&member("ollama:llama3.2:latest"), &overrides).unwrap();
        assert_eq!(provider.options().timeout, Duration::from_secs(321));
    }
}
