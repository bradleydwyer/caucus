//! `caucus review`: adjudicated multi-model code/text review.
//!
//! A narrow, fixed four-phase pipeline — not a generic DAG:
//!
//! 1. **Review** — every council member independently reviews the input and
//!    returns findings under a strict JSON schema (direct JSON or fenced
//!    blocks only; prose is never accepted as findings).
//! 2. **Anonymize** — findings are assigned deterministic anonymous source
//!    IDs so peer voting is blind.
//! 3. **Vote** — every member casts schema-level support/oppose/abstain
//!    votes with evidence-linked reasons over all anonymized findings.
//! 4. **Adjudicate** — a council judge accepts or rejects each finding with
//!    a reason citing the evidence and the raw votes.
//!
//! Accepted findings are classified `unanimous` / `majority` / `disputed`
//! from the raw votes: unanimous only when every valid ballot supports,
//! majority only when more than half of all valid ballots support. Partial
//! failures are warnings; quorum is enforced on semantically valid parsed
//! results in every LLM phase — transport successes that fail parsing or
//! ballot validation never count. Parse failures are recorded honestly —
//! votes and scores are never fabricated. Each phase checkpoints to a narrow
//! versioned JSON state file (written atomically via temp file + rename);
//! `--resume` validates the input/options hash and reuses completed phases.
//! Git input is read via `ProcessSpec`/`run_argv` with an explicit argv,
//! never a shell.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use caucus_core::provider::{FanoutReport, MultiProvider, fanout};
use caucus_core::{
    Config, Council, FanoutConfig, LlmProvider, ProcessLimits, ProcessOutput, ProcessSpec,
    Transport, run_argv,
};
use clap::Args;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::commands::council;
use crate::commands::run::RunDeadline;

// ---------------------------------------------------------------------------
// Deterministic FNV-1a 64-bit hashing (local, no new dependency)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Lowercase hex rendering of [`fnv64`], zero-padded to 16 digits.
pub fn fnv64_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv64(bytes))
}

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Args)]
pub struct ReviewArgs {
    /// Review a single file (conflicts with --git-diff/--staged; default is stdin)
    #[arg(long, conflicts_with_all = ["git_diff", "staged"])]
    pub file: Option<PathBuf>,

    /// Review the working-tree diff (`git diff`)
    #[arg(long, conflicts_with = "staged")]
    pub git_diff: bool,

    /// Review the staged diff (`git diff --staged`)
    #[arg(long)]
    pub staged: bool,

    /// Council profile name (defaults to configured default, then `deep`)
    #[arg(long, conflicts_with = "auto")]
    pub profile: Option<String>,

    /// Use the zero-key auto council from locally discovered adapters
    #[arg(long)]
    pub auto: bool,

    /// Path to config file
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Receipt format
    #[arg(long, default_value = "markdown", value_parser = ["markdown", "json"])]
    pub format: String,

    /// Write the receipt to a file instead of stdout
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Write the provenance manifest (full receipt JSON) to this path;
    /// the checkpoint state lives next to it
    #[arg(long)]
    pub manifest: Option<PathBuf>,

    /// Resume from the checkpoint state, reusing completed phases
    #[arg(long)]
    pub resume: bool,

    /// Minimum successful participants per LLM phase (default: profile quorum)
    #[arg(long)]
    pub quorum: Option<usize>,

    /// Whole-run wall-clock deadline in seconds (default: profile deadline)
    #[arg(long)]
    pub deadline_secs: Option<u64>,

    /// Per-provider request/process timeout in seconds
    #[arg(long)]
    pub request_timeout_secs: Option<u64>,

    /// Maximum concurrent requests (default: 4)
    #[arg(long)]
    pub max_concurrency: Option<usize>,

    /// Advisory budget in USD (recorded; not enforced — transports expose no cost data)
    #[arg(long)]
    pub budget_usd: Option<f64>,

    /// Hard cap on total provider requests across all phases
    #[arg(long)]
    pub max_requests: Option<usize>,
}

// ---------------------------------------------------------------------------
// Strict finding / vote / adjudication schemas
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        };
        f.write_str(s)
    }
}

/// One finding as returned by a reviewing member. Strict: unknown fields and
/// missing required fields are parse errors, never silently tolerated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawFinding {
    pub title: String,
    pub severity: Severity,
    pub evidence: String,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub line: Option<u64>,
    #[serde(default)]
    pub recommendation: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FindingsDoc {
    findings: Vec<RawFinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vote {
    Support,
    Oppose,
    Abstain,
}

impl std::fmt::Display for Vote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Vote::Support => "support",
            Vote::Oppose => "oppose",
            Vote::Abstain => "abstain",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteCast {
    pub finding_id: String,
    pub vote: Vote,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VotesDoc {
    votes: Vec<VoteCast>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjudication {
    pub finding_id: String,
    pub accepted: bool,
    pub reason: String,
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdjudicationDoc {
    adjudications: Vec<Adjudication>,
}

/// Accepted-finding classification derived from raw votes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Classification {
    Unanimous,
    Majority,
    Disputed,
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Classification::Unanimous => "unanimous",
            Classification::Majority => "majority",
            Classification::Disputed => "disputed",
        };
        f.write_str(s)
    }
}

/// Classify from counted votes. `ballots` is the number of valid ballots
/// cast; unparseable or schema-incomplete ballots are excluded upstream,
/// never counted as abstain. Unanimous requires every valid ballot to
/// support; majority requires support from more than half of all valid
/// ballots (abstentions and omissions count as non-support); anything else
/// is disputed.
pub fn classify(support: usize, ballots: usize) -> Classification {
    if ballots > 0 && support == ballots {
        Classification::Unanimous
    } else if support * 2 > ballots {
        Classification::Majority
    } else {
        Classification::Disputed
    }
}

// ---------------------------------------------------------------------------
// JSON extraction: direct documents and fenced blocks only — never prose
// ---------------------------------------------------------------------------

/// Candidate JSON texts: the whole trimmed response, then each fenced code
/// block. Anything outside a fence is prose and is never parsed as findings.
fn json_candidates(text: &str) -> Vec<String> {
    let mut candidates = vec![text.trim().to_string()];
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after_open = &rest[start + 3..];
        // Skip an optional language tag on the opening fence line.
        let body_start = after_open.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after_open[body_start..];
        match body.find("```") {
            Some(end) => {
                candidates.push(body[..end].trim().to_string());
                rest = &body[end + 3..];
            }
            None => break,
        }
    }
    candidates
}

/// Parse member findings: a `{"findings": [...]}` document or a bare array,
/// either as the whole response or inside a fenced block. Prose-only
/// responses are an honest error.
pub fn parse_findings(text: &str) -> std::result::Result<Vec<RawFinding>, String> {
    for candidate in json_candidates(text) {
        if let Ok(doc) = serde_json::from_str::<FindingsDoc>(&candidate) {
            return Ok(doc.findings);
        }
        if let Ok(array) = serde_json::from_str::<Vec<RawFinding>>(&candidate) {
            return Ok(array);
        }
    }
    Err("response contained no strict or fenced JSON findings document".to_string())
}

/// Parse one ballot: `{"votes": [...]}` only (direct or fenced).
pub fn parse_votes(text: &str) -> std::result::Result<Vec<VoteCast>, String> {
    for candidate in json_candidates(text) {
        if let Ok(doc) = serde_json::from_str::<VotesDoc>(&candidate) {
            return Ok(doc.votes);
        }
    }
    Err("response contained no strict or fenced JSON votes document".to_string())
}

/// Parse judge adjudications: `{"adjudications": [...]}` only (direct or fenced).
pub fn parse_adjudications(text: &str) -> std::result::Result<Vec<Adjudication>, String> {
    for candidate in json_candidates(text) {
        if let Ok(doc) = serde_json::from_str::<AdjudicationDoc>(&candidate) {
            return Ok(doc.adjudications);
        }
    }
    Err("judge response contained no strict or fenced JSON adjudications document".to_string())
}

fn validate_adjudications(
    adjudications: &[Adjudication],
    findings: &[AttributedFinding],
) -> std::result::Result<(), String> {
    let known: std::collections::HashSet<&str> =
        findings.iter().map(|finding| finding.id.as_str()).collect();
    let mut seen = std::collections::HashSet::new();
    for adjudication in adjudications {
        if !known.contains(adjudication.finding_id.as_str()) {
            return Err(format!("unknown finding_id `{}`", adjudication.finding_id));
        }
        if !seen.insert(adjudication.finding_id.as_str()) {
            return Err(format!("duplicate finding_id `{}`", adjudication.finding_id));
        }
    }
    let missing: Vec<&str> = findings
        .iter()
        .map(|finding| finding.id.as_str())
        .filter(|id| !seen.contains(id))
        .collect();
    if !missing.is_empty() {
        return Err(format!("missing adjudications for {}", missing.join(", ")));
    }
    Ok(())
}

/// Deterministic anonymous source ID for a council member.
pub fn anon_source_id(member: &str) -> String {
    format!("src-{}", &fnv64_hex(member.as_bytes())[..8])
}

// ---------------------------------------------------------------------------
// Input acquisition
// ---------------------------------------------------------------------------

/// Build the explicit git argv for diff input. Never a shell.
pub fn git_diff_argv(staged: bool) -> Vec<String> {
    let mut argv = vec!["git".to_string(), "diff".to_string()];
    if staged {
        argv.push("--staged".to_string());
    }
    argv
}

async fn read_git_diff(staged: bool) -> Result<String> {
    let mut spec = ProcessSpec::new("git")
        .arg("diff")
        .inherit_env("GIT_DIR")
        .inherit_env("GIT_WORK_TREE")
        .inherit_env("GIT_INDEX_FILE")
        .inherit_env("GIT_OBJECT_DIRECTORY")
        .inherit_env("GIT_ALTERNATE_OBJECT_DIRECTORIES");
    if staged {
        spec = spec.arg("--staged");
    }
    let output = run_argv(&spec, &ProcessLimits::default())
        .await
        .with_context(|| format!("failed to run `{}`", git_diff_argv(staged).join(" ")))?;
    complete_git_diff(output, &git_diff_argv(staged).join(" "))
}

fn complete_git_diff(output: ProcessOutput, command: &str) -> Result<String> {
    if !output.success() {
        anyhow::bail!("`{command}` exited with {:?}: {}", output.exit_code, output.stderr.trim());
    }
    if output.truncated {
        anyhow::bail!(
            "`{command}` exceeded the {}-byte capture limit; refusing to review an incomplete diff",
            ProcessLimits::default().max_stdout_bytes
        );
    }
    Ok(output.stdout)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMeta {
    pub source: String,
    pub detail: String,
    pub bytes: usize,
    pub hash: String,
}

async fn acquire_input(args: &ReviewArgs) -> Result<(String, InputMeta)> {
    let (content, source, detail) = if let Some(path) = &args.file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        (content, "file", path.display().to_string())
    } else if args.git_diff {
        (read_git_diff(false).await?, "git-diff", git_diff_argv(false).join(" "))
    } else if args.staged {
        (read_git_diff(true).await?, "git-staged", git_diff_argv(true).join(" "))
    } else {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
            .context("failed to read stdin")?;
        (buf, "stdin", "-".to_string())
    };
    if content.trim().is_empty() {
        anyhow::bail!("nothing to review: {source} input is empty");
    }
    let meta =
        InputMeta { source: source.to_string(), detail, bytes: content.len(), hash: String::new() };
    Ok((content, meta))
}

// ---------------------------------------------------------------------------
// Request budget enforcement (max_requests is a hard cap)
// ---------------------------------------------------------------------------

/// Wraps a provider with a shared request counter. Every admitted request
/// increments the counter — capped or not — so the counter is always the
/// receipt's request total. Requests beyond the cap fail honestly and are
/// rolled back, never counted.
struct BudgetedProvider {
    inner: Arc<dyn LlmProvider>,
    used: Arc<AtomicUsize>,
    max: Option<usize>,
}

#[async_trait::async_trait]
impl LlmProvider for BudgetedProvider {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let prev = self.used.fetch_add(1, Ordering::SeqCst);
        if let Some(max) = self.max
            && prev >= max
        {
            self.used.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("request budget exhausted (--max-requests {max})");
        }
        self.inner.complete(prompt, system).await
    }

    fn transport(&self) -> Transport {
        self.inner.transport()
    }

    fn options(&self) -> caucus_core::ProviderOptions {
        self.inner.options()
    }
}

fn wrap_with_budget(
    base: &MultiProvider,
    used: Arc<AtomicUsize>,
    max: Option<usize>,
) -> MultiProvider {
    let mut multi = MultiProvider::new();
    for (name, provider) in base.iter() {
        let wrapped =
            BudgetedProvider { inner: Arc::clone(provider), used: Arc::clone(&used), max };
        multi = multi.add_shared(name.to_string(), Arc::new(wrapped));
    }
    multi
}

/// Wrap the judge's provider in the shared request budget. The judge is
/// selected from the *unwrapped* council providers — or built dedicated when
/// the designated judge is not a council member — so every adjudication
/// request consumes the same hard budget as council members, exactly once.
fn budgeted_judge(
    base: &MultiProvider,
    name: &str,
    judge: council::JudgeProvider<'_>,
    used: Arc<AtomicUsize>,
    max: Option<usize>,
) -> BudgetedProvider {
    let inner: Arc<dyn LlmProvider> = match judge {
        council::JudgeProvider::Borrowed(_) => {
            base.get_shared(name).expect("borrowed judge is a registered council member")
        }
        council::JudgeProvider::Owned(provider) => Arc::from(provider),
    };
    BudgetedProvider { inner, used, max }
}

async fn complete_with_timeout(
    provider: &dyn LlmProvider,
    prompt: &str,
    system: Option<&str>,
    timeout: Duration,
    label: &str,
) -> Result<String> {
    match tokio::time::timeout(timeout, provider.complete(prompt, system)).await {
        Ok(result) => result,
        Err(_) => Err(caucus_core::ProviderError::timeout(format!(
            "{label} exceeded {}s request timeout",
            timeout.as_secs()
        ))
        .into()),
    }
}

// ---------------------------------------------------------------------------
// Checkpoint state (narrow, versioned)
// ---------------------------------------------------------------------------

pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantRecord {
    pub member: String,
    pub transport: String,
    /// "ok" | "failed" | "parse-error" | "incomplete-ballot"
    pub status: String,
    pub latency_ms: u64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributedFinding {
    /// Global deterministic ID (F1, F2, ...) in collection order.
    pub id: String,
    /// Anonymous source ID shown to peer voters.
    pub source: String,
    /// Originating member (provenance; hidden from voters).
    pub member: String,
    pub finding: RawFinding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ballot {
    pub voter: String,
    pub votes: Vec<VoteCast>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Phase1State {
    participants: Vec<ParticipantRecord>,
    findings: Vec<AttributedFinding>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Phase3State {
    participants: Vec<ParticipantRecord>,
    ballots: Vec<Ballot>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Phase4State {
    judge: ParticipantRecord,
    adjudications: Vec<Adjudication>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub run_id: String,
    pub input_hash: String,
    pub options_hash: String,
    /// Provider requests admitted so far across all phases; the resume seed.
    #[serde(default)]
    pub requests_used: usize,
    #[serde(default)]
    phase1: Option<Phase1State>,
    #[serde(default)]
    phase3: Option<Phase3State>,
    #[serde(default)]
    phase4: Option<Phase4State>,
}

impl Checkpoint {
    fn fresh(run_id: String, input_hash: String, options_hash: String) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            run_id,
            input_hash,
            options_hash,
            requests_used: 0,
            phase1: None,
            phase3: None,
            phase4: None,
        }
    }
}

/// Checkpoint lives next to the manifest, or in the cwd by default.
pub fn checkpoint_path(manifest: Option<&Path>) -> PathBuf {
    match manifest {
        Some(p) => {
            let mut s = p.as_os_str().to_owned();
            s.push(".checkpoint.json");
            PathBuf::from(s)
        }
        None => PathBuf::from(".caucus-review.checkpoint.json"),
    }
}

/// Write `contents` to `path` atomically: write a sibling temp file, then
/// rename it over `path`. The rename is atomic on the same filesystem, so an
/// interruption can leave at most a stale temp file — never a partially
/// written resume/manifest file at `path`.
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(format!(".tmp-{}", std::process::id()));
    let tmp = PathBuf::from(tmp_os);
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)
}

fn save_checkpoint(path: &Path, checkpoint: &Checkpoint) -> Result<()> {
    let json = serde_json::to_string_pretty(checkpoint)?;
    write_atomic(path, &json)
        .with_context(|| format!("failed to write checkpoint {}", path.display()))
}

fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("--resume given but no checkpoint at {}", path.display()))?;
    let checkpoint: Checkpoint = serde_json::from_str(&text)
        .with_context(|| format!("checkpoint {} is not valid state JSON", path.display()))?;
    if checkpoint.version != CHECKPOINT_VERSION {
        anyhow::bail!(
            "checkpoint version {} is not supported (expected {CHECKPOINT_VERSION})",
            checkpoint.version
        );
    }
    Ok(checkpoint)
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MemberEntry {
    pub member: String,
    pub utility: String,
    pub model: String,
    pub effort: Option<String>,
    pub transport: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ManifestSection {
    pub profile: String,
    pub options: OptionsSection,
    pub members: Vec<MemberEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OptionsSection {
    pub quorum: usize,
    pub deadline_secs: Option<u64>,
    pub request_timeout_secs: Option<u64>,
    pub max_concurrency: usize,
    pub budget_usd: Option<f64>,
    pub max_requests: Option<usize>,
    pub options_hash: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptVote {
    pub voter: String,
    pub vote: Vote,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptFinding {
    pub id: String,
    pub title: String,
    pub severity: Severity,
    pub file: Option<String>,
    pub line: Option<u64>,
    pub evidence: String,
    pub recommendation: Option<String>,
    pub source: String,
    pub member: String,
    pub votes: Vec<ReceiptVote>,
    pub dissent: Vec<ReceiptVote>,
    pub adjudication: Option<Adjudication>,
    pub classification: Option<Classification>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhasesSection {
    pub review: Vec<ParticipantRecord>,
    pub voting: Vec<ParticipantRecord>,
    pub adjudication: Option<ParticipantRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestsSection {
    pub used: usize,
    pub max: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewReceipt {
    pub schema: u32,
    pub run_id: String,
    pub input: InputMeta,
    pub manifest: ManifestSection,
    pub phases: PhasesSection,
    pub findings: Vec<ReceiptFinding>,
    pub warnings: Vec<String>,
    pub budget_usd: Option<f64>,
    pub budget_note: Option<String>,
    pub deadline_secs: Option<u64>,
    pub request_timeout_secs: Option<u64>,
    pub requests: RequestsSection,
    pub resumed: bool,
}

pub fn render_markdown(receipt: &ReviewReceipt) -> String {
    let mut out = String::new();
    out.push_str("# Caucus Review Receipt\n\n");
    out.push_str(&format!("- Run: `{}`\n", receipt.run_id));
    out.push_str(&format!(
        "- Input: {} (`{}`), {} bytes, hash `{}`\n",
        receipt.input.source, receipt.input.detail, receipt.input.bytes, receipt.input.hash
    ));
    out.push_str(&format!(
        "- Profile: `{}` (quorum {}, concurrency {})\n",
        receipt.manifest.profile,
        receipt.manifest.options.quorum,
        receipt.manifest.options.max_concurrency
    ));
    if let Some(deadline) = receipt.deadline_secs {
        out.push_str(&format!("- Whole-run deadline: {deadline}s\n"));
    }
    if let Some(timeout) = receipt.request_timeout_secs {
        out.push_str(&format!("- Per-request timeout: {timeout}s\n"));
    }
    match (receipt.budget_usd, &receipt.budget_note) {
        (Some(b), _) => out.push_str(&format!("- Budget: ${b:.2} (advisory — not enforced)\n")),
        (None, Some(note)) => out.push_str(&format!("- Budget: {note}\n")),
        (None, None) => {}
    }
    match receipt.requests.max {
        Some(max) => {
            out.push_str(&format!("- Requests: {}/{max} used\n", receipt.requests.used));
        }
        None => out.push_str(&format!("- Requests: {} used (no cap)\n", receipt.requests.used)),
    }
    if receipt.resumed {
        out.push_str("- Resumed: yes (completed phases reused from checkpoint)\n");
    }

    out.push_str("\n## Members\n\n");
    for m in &receipt.manifest.members {
        let effort = m.effort.as_deref().unwrap_or("-");
        out.push_str(&format!(
            "- `{}` — utility `{}`, model `{}`, effort `{}`, transport `{}`\n",
            m.member, m.utility, m.model, effort, m.transport
        ));
    }

    out.push_str("\n## Phase participants\n\n");
    for (name, records) in [("review", &receipt.phases.review), ("voting", &receipt.phases.voting)]
    {
        out.push_str(&format!("### {name}\n\n"));
        for p in records {
            match &p.error {
                Some(err) => out.push_str(&format!(
                    "- `{}`: {} ({} ms) — {}\n",
                    p.member, p.status, p.latency_ms, err
                )),
                None => {
                    out.push_str(&format!("- `{}`: {} ({} ms)\n", p.member, p.status, p.latency_ms))
                }
            }
        }
        out.push('\n');
    }
    if let Some(judge) = &receipt.phases.adjudication {
        out.push_str("### adjudication\n\n");
        match &judge.error {
            Some(err) => out.push_str(&format!(
                "- `{}`: {} ({} ms) — {}\n",
                judge.member, judge.status, judge.latency_ms, err
            )),
            None => out.push_str(&format!(
                "- `{}`: {} ({} ms)\n",
                judge.member, judge.status, judge.latency_ms
            )),
        }
    }

    out.push_str("\n## Findings\n\n");
    if receipt.findings.is_empty() {
        out.push_str("No findings.\n");
    }
    for f in &receipt.findings {
        let location = match (&f.file, f.line) {
            (Some(file), Some(line)) => format!(" ({file}:{line})"),
            (Some(file), None) => format!(" ({file})"),
            (None, _) => String::new(),
        };
        let verdict = match (&f.adjudication, f.classification) {
            (Some(a), Some(c)) if a.accepted => format!("accepted — {c}"),
            (Some(a), None) if !a.accepted => "rejected".to_string(),
            _ => "not adjudicated".to_string(),
        };
        out.push_str(&format!("### {} [{}]{} — {}\n\n", f.id, f.severity, location, verdict));
        out.push_str(&format!("{}\n\n", f.title));
        out.push_str(&format!("- Evidence: {}\n", f.evidence));
        if let Some(rec) = &f.recommendation {
            out.push_str(&format!("- Recommendation: {rec}\n"));
        }
        out.push_str(&format!("- Source: `{}` (member `{}`)\n", f.source, f.member));
        if let Some(a) = &f.adjudication {
            out.push_str(&format!("- Adjudication: {}\n", a.reason));
            if let Some(ev) = &a.evidence {
                out.push_str(&format!("- Adjudication evidence: {ev}\n"));
            }
        }
        if !f.votes.is_empty() {
            out.push_str("- Votes:\n");
            for v in &f.votes {
                let reason = v.reason.as_deref().unwrap_or("-");
                out.push_str(&format!("  - `{}`: {} — {}\n", v.voter, v.vote, reason));
            }
        }
        if !f.dissent.is_empty() {
            out.push_str("- Dissent:\n");
            for v in &f.dissent {
                let reason = v.reason.as_deref().unwrap_or("-");
                out.push_str(&format!("  - `{}`: {} — {}\n", v.voter, v.vote, reason));
            }
        }
        out.push('\n');
    }

    if !receipt.warnings.is_empty() {
        out.push_str("## Warnings\n\n");
        for w in &receipt.warnings {
            out.push_str(&format!("- {w}\n"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const REVIEW_SYSTEM: &str = "You are a meticulous code reviewer. Respond with ONLY a JSON object \
of the form {\"findings\": [{\"title\": string, \"severity\": \"low\"|\"medium\"|\"high\"|\"critical\", \
\"evidence\": string, \"file\": string|null, \"line\": number|null, \"recommendation\": string|null}]}. \
Evidence must quote or precisely reference the reviewed material. No prose, no markdown fences. \
If there are no findings, return {\"findings\": []}.";

const VOTE_SYSTEM: &str = "You are a blind peer reviewer voting on anonymized findings. For each \
finding, cast exactly one vote: \"support\", \"oppose\", or \"abstain\". Respond with ONLY a JSON \
object {\"votes\": [{\"finding_id\": string, \"vote\": \"support\"|\"oppose\"|\"abstain\", \
\"reason\": string}]}. Reasons must cite the finding's evidence. No prose, no markdown fences.";

const JUDGE_SYSTEM: &str = "You are the adjudicating judge of a code review council. You receive \
findings and the raw anonymized peer votes. For each finding decide accepted (true/false) with a \
reason that cites the finding's evidence and the votes. Respond with ONLY a JSON object \
{\"adjudications\": [{\"finding_id\": string, \"accepted\": boolean, \"reason\": string, \
\"evidence\": string|null}]}. No prose, no markdown fences.";

fn review_prompt(source_detail: &str, content: &str) -> String {
    format!(
        "Review the following material from {source_detail}. Report only concrete, \
         evidence-backed defects, risks, or correctness issues.\n\n\
         --- BEGIN MATERIAL ---\n{content}\n--- END MATERIAL ---"
    )
}

fn vote_prompt(findings: &[AttributedFinding]) -> String {
    let anonymized: Vec<serde_json::Value> = findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "source": f.source,
                "title": f.finding.title,
                "severity": f.finding.severity,
                "file": f.finding.file,
                "line": f.finding.line,
                "evidence": f.finding.evidence,
                "recommendation": f.finding.recommendation,
            })
        })
        .collect();
    format!(
        "Vote on every finding below. You must cast one vote per finding ID.\n\n{}",
        serde_json::to_string_pretty(&anonymized).expect("findings serialize")
    )
}

fn judge_prompt(findings: &[AttributedFinding], ballots: &[Ballot]) -> String {
    let anonymized: Vec<serde_json::Value> = findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "source": f.source,
                "title": f.finding.title,
                "severity": f.finding.severity,
                "file": f.finding.file,
                "line": f.finding.line,
                "evidence": f.finding.evidence,
            })
        })
        .collect();
    let votes: Vec<serde_json::Value> = ballots
        .iter()
        .map(|b| {
            serde_json::json!({
                "voter": anon_source_id(&b.voter),
                "votes": b.votes,
            })
        })
        .collect();
    format!(
        "Adjudicate every finding. Findings:\n{}\n\nRaw votes:\n{}",
        serde_json::to_string_pretty(&anonymized).expect("findings serialize"),
        serde_json::to_string_pretty(&votes).expect("votes serialize"),
    )
}

// ---------------------------------------------------------------------------
// Phase helpers
// ---------------------------------------------------------------------------

/// A transport-successful response that did not yield a semantically valid
/// phase result: unparseable JSON, or a schema-incomplete ballot. Recorded
/// honestly; never counted toward quorum, votes, or classification.
#[derive(Debug)]
struct InvalidResponse {
    model: String,
    /// Participant status: "parse-error" | "incomplete-ballot".
    status: &'static str,
    error: String,
}

fn participant_records(
    report: &FanoutReport,
    invalid: &[InvalidResponse],
) -> Vec<ParticipantRecord> {
    let mut records: Vec<ParticipantRecord> = report
        .successes
        .iter()
        .map(|s| {
            let problem = invalid.iter().find(|i| i.model == s.model);
            ParticipantRecord {
                member: s.model.clone(),
                transport: s.transport.to_string(),
                status: problem.map(|i| i.status.to_string()).unwrap_or_else(|| "ok".to_string()),
                latency_ms: s.latency_ms,
                error: problem.map(|i| i.error.clone()),
            }
        })
        .collect();
    records.extend(report.failures.iter().map(|f| ParticipantRecord {
        member: f.model.clone(),
        transport: f.transport.to_string(),
        status: "failed".to_string(),
        latency_ms: f.latency_ms,
        error: Some(format!("{}: {}", f.kind, f.message)),
    }));
    records
}

/// Quorum is semantic: only responses that parsed into a valid phase result
/// count — a findings document (possibly with zero findings) in review, a
/// complete ballot in voting. Transport successes that fail parsing or
/// ballot validation never satisfy quorum.
fn enforce_quorum(phase: &str, valid: usize, quorum: usize) -> Result<()> {
    if valid >= quorum {
        return Ok(());
    }
    anyhow::bail!(
        "quorum not met in {phase} phase: {valid}/{quorum} required member(s) produced a valid response"
    )
}

/// Parse every transport-successful review response into attributed findings.
/// Returns the findings, the number of semantically valid responses (a valid
/// document with zero findings counts), the invalid responses with
/// phase-correct errors, and per-response warnings.
fn collect_findings(
    report: &FanoutReport,
) -> (Vec<AttributedFinding>, usize, Vec<InvalidResponse>, Vec<String>) {
    let mut findings: Vec<AttributedFinding> = Vec::new();
    let mut invalid: Vec<InvalidResponse> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for success in &report.successes {
        match parse_findings(&success.content) {
            Ok(raw) => {
                for finding in raw {
                    let id = format!("F{}", findings.len() + 1);
                    findings.push(AttributedFinding {
                        id,
                        source: anon_source_id(&success.model),
                        member: success.model.clone(),
                        finding,
                    });
                }
            }
            Err(e) => {
                invalid.push(InvalidResponse {
                    model: success.model.clone(),
                    status: "parse-error",
                    error: format!("response was not valid findings JSON: {e}"),
                });
                warnings.push(format!(
                    "review: {} returned no valid findings JSON: {e}",
                    success.model
                ));
            }
        }
    }
    let valid = report.successes.len() - invalid.len();
    (findings, valid, invalid, warnings)
}

/// Validate a parsed ballot against the known finding IDs: exactly one vote
/// per known finding, no duplicates, no unknown IDs. Returns the votes in
/// cast order on success, or the list of problems on failure.
fn validate_ballot(
    votes: &[VoteCast],
    known: &std::collections::HashSet<&str>,
) -> std::result::Result<Vec<VoteCast>, Vec<String>> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut problems: Vec<String> = Vec::new();
    let mut valid: Vec<VoteCast> = Vec::new();
    for vote in votes {
        if !known.contains(vote.finding_id.as_str()) {
            problems.push(format!("vote for unknown finding `{}`", vote.finding_id));
        } else if !seen.insert(vote.finding_id.as_str()) {
            problems.push(format!("duplicate vote for `{}`", vote.finding_id));
        } else {
            valid.push(vote.clone());
        }
    }
    let mut missing: Vec<&str> = known.iter().copied().filter(|id| !seen.contains(id)).collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        problems.push(format!("missing vote(s) for {}", missing.join(", ")));
    }
    if problems.is_empty() { Ok(valid) } else { Err(problems) }
}

/// Parse every transport-successful voting response into a ballot. A ballot
/// is valid only when it casts exactly one vote for every known finding — no
/// missing findings, no duplicates, no unknown IDs. Invalid ballots are
/// recorded honestly and never counted toward quorum or classification.
fn collect_ballots(
    report: &FanoutReport,
    known: &std::collections::HashSet<&str>,
) -> (Vec<Ballot>, Vec<InvalidResponse>, Vec<String>) {
    let mut ballots: Vec<Ballot> = Vec::new();
    let mut invalid: Vec<InvalidResponse> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for success in &report.successes {
        match parse_votes(&success.content) {
            Ok(votes) => match validate_ballot(&votes, known) {
                Ok(valid) => {
                    ballots.push(Ballot { voter: success.model.clone(), votes: valid });
                }
                Err(problems) => {
                    invalid.push(InvalidResponse {
                        model: success.model.clone(),
                        status: "incomplete-ballot",
                        error: format!("ballot discarded: {}", problems.join("; ")),
                    });
                    warnings.push(format!(
                        "voting: {} cast an incomplete ballot ({}); discarded",
                        success.model,
                        problems.join("; ")
                    ));
                }
            },
            Err(e) => {
                // Honest handling: the ballot is missing, never fabricated.
                invalid.push(InvalidResponse {
                    model: success.model.clone(),
                    status: "parse-error",
                    error: format!("response was not valid votes JSON: {e}"),
                });
                warnings
                    .push(format!("voting: {} returned no valid votes JSON: {e}", success.model));
            }
        }
    }
    (ballots, invalid, warnings)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(args: ReviewArgs) -> Result<()> {
    let (_path, config) = council::load_config(args.config.as_deref())?;

    // Profile: explicit flag, then configured default, then built-in `deep`.
    let mut council: Council = if args.auto {
        let selection = council::auto_council(&config).await;
        for exclusion in &selection.exclusions {
            eprintln!("  {} {} excluded: {}", "·".dimmed(), exclusion.adapter, exclusion.reason);
        }
        selection.council
    } else if let Some(name) = args.profile.as_deref() {
        config.resolve_profile(Some(name))?
    } else {
        match config.resolve_profile(None) {
            Ok(c) => c,
            Err(_) => config.resolve_profile(Some("deep"))?,
        }
    };
    if council.members.is_empty() {
        anyhow::bail!(
            "council '{}' is empty — no members to run a review; run `caucus doctor` for details",
            council.name
        );
    }

    let quorum = args.quorum.unwrap_or(council.quorum);
    if quorum == 0 || quorum > council.members.len() {
        anyhow::bail!(
            "quorum {quorum} is invalid for council '{}' with {} member(s)",
            council.name,
            council.members.len()
        );
    }
    let deadline_secs = args.deadline_secs.or(council.deadline_secs);
    let request_timeout_secs =
        args.request_timeout_secs.or(council.request_timeout_secs).or(deadline_secs);
    if deadline_secs == Some(0) {
        anyhow::bail!("--deadline-secs must be at least 1");
    }
    if request_timeout_secs == Some(0) {
        anyhow::bail!("--request-timeout-secs must be at least 1");
    }
    // Provider construction must see CLI-level overrides too, not only the
    // original profile values. Outer fan-out/judge timeouts remain the final
    // cancellation boundary.
    council.deadline_secs = deadline_secs;
    council.request_timeout_secs = request_timeout_secs;
    let max_concurrency = args.max_concurrency.unwrap_or(4).max(1);
    let budget_usd = args.budget_usd.or(council.budget_usd);
    let max_requests = args.max_requests;

    let (content, mut input) = acquire_input(&args).await?;
    input.hash = fnv64_hex(content.as_bytes());

    // Deterministic run identity: input + resolved options.
    let members: Vec<String> = council.members.iter().map(|m| m.to_string()).collect();
    let fingerprint = serde_json::json!({
        "profile": council.name,
        "members": members,
        "quorum": quorum,
        "deadline_secs": deadline_secs,
        "request_timeout_secs": request_timeout_secs,
        "max_concurrency": max_concurrency,
        "budget_usd": budget_usd,
        "max_requests": max_requests,
    });
    let options_hash = fnv64_hex(fingerprint.to_string().as_bytes());
    let run_id = fnv64_hex(format!("{}:{}", input.hash, options_hash).as_bytes());

    let ckpt_path = checkpoint_path(args.manifest.as_deref());
    let mut checkpoint = if args.resume {
        let loaded = load_checkpoint(&ckpt_path)?;
        if loaded.input_hash != input.hash || loaded.options_hash != options_hash {
            anyhow::bail!(
                "checkpoint {} does not match this run (input or options changed); \
                 refusing to resume",
                ckpt_path.display()
            );
        }
        eprintln!(
            "{} Resuming run {} from {}",
            "▶".green(),
            loaded.run_id.cyan(),
            ckpt_path.display()
        );
        loaded
    } else {
        Checkpoint::fresh(run_id.clone(), input.hash.clone(), options_hash.clone())
    };
    let resumed = args.resume;
    let run_deadline = RunDeadline::new(deadline_secs);

    let base = council::build_council_provider(&council, &config)?;
    // A resumed run continues the checkpoint's cumulative request total so
    // the hard cap spans both runs and the receipt reports the true total.
    let request_count = Arc::new(AtomicUsize::new(checkpoint.requests_used));
    let multi = wrap_with_budget(&base, Arc::clone(&request_count), max_requests);
    let requested_timeout = Duration::from_secs(
        request_timeout_secs.unwrap_or(caucus_core::DEFAULT_REQUEST_TIMEOUT.as_secs()),
    );

    let mut warnings: Vec<String> = Vec::new();
    if budget_usd.is_some() {
        warnings.push(
            "budget_usd is advisory: current transports expose no cost data; recorded but not enforced"
                .to_string(),
        );
    }

    eprintln!(
        "{} Review run {} — council '{}' ({} member(s), quorum {})",
        "▶".green(),
        run_id.cyan(),
        council.name.cyan(),
        council.members.len(),
        quorum,
    );

    // -- Phase 1+2: independent reviews, then deterministic anonymization.
    let phase1 = match checkpoint.phase1.clone() {
        Some(state) => {
            eprintln!("  {} review phase reused from checkpoint", "·".dimmed());
            state
        }
        None => {
            let timeout = run_deadline.turn_timeout(requested_timeout)?;
            let report = run_deadline
                .wait(fanout(
                    &multi,
                    &review_prompt(&input.detail, &content),
                    Some(REVIEW_SYSTEM),
                    FanoutConfig { max_concurrency, timeout, quorum },
                ))
                .await?;
            for warning in report.warnings() {
                eprintln!("  {} {}", "✗".red(), warning);
                warnings.push(format!("review: {warning}"));
            }
            let (findings, valid, invalid, parse_warnings) = collect_findings(&report);
            warnings.extend(parse_warnings);
            for i in &invalid {
                eprintln!("  {} {} — {}", "✗".red(), i.model, i.error);
            }
            // Quorum counts semantically valid responses only: unparseable
            // responses never satisfy it, an empty findings document does.
            enforce_quorum("review", valid, quorum)?;

            let state = Phase1State {
                participants: participant_records(&report, &invalid),
                findings,
                warnings: warnings.clone(),
            };
            checkpoint.phase1 = Some(state.clone());
            checkpoint.requests_used = request_count.load(Ordering::SeqCst);
            save_checkpoint(&ckpt_path, &checkpoint)?;
            eprintln!(
                "  {} review phase: {} finding(s) from {} valid response(s)",
                "✓".green(),
                state.findings.len(),
                valid
            );
            state
        }
    };
    for w in phase1.warnings.clone() {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }

    // -- Phase 3: blind peer voting (skipped when there is nothing to vote on).
    let phase3 = if phase1.findings.is_empty() {
        warnings.push("no findings produced; voting and adjudication skipped".to_string());
        None
    } else {
        let state = match checkpoint.phase3.clone() {
            Some(state) => {
                eprintln!("  {} voting phase reused from checkpoint", "·".dimmed());
                state
            }
            None => {
                let timeout = run_deadline.turn_timeout(requested_timeout)?;
                let report = run_deadline
                    .wait(fanout(
                        &multi,
                        &vote_prompt(&phase1.findings),
                        Some(VOTE_SYSTEM),
                        FanoutConfig { max_concurrency, timeout, quorum },
                    ))
                    .await?;
                for warning in report.warnings() {
                    eprintln!("  {} {}", "✗".red(), warning);
                    warnings.push(format!("voting: {warning}"));
                }
                let known: std::collections::HashSet<&str> =
                    phase1.findings.iter().map(|f| f.id.as_str()).collect();
                let (ballots, invalid, parse_warnings) = collect_ballots(&report, &known);
                warnings.extend(parse_warnings);
                for i in &invalid {
                    eprintln!("  {} {} — {}", "✗".red(), i.model, i.error);
                }
                // Quorum counts complete ballots only: unparseable or
                // schema-incomplete ballots never satisfy it.
                enforce_quorum("voting", ballots.len(), quorum)?;

                let state = Phase3State {
                    participants: participant_records(&report, &invalid),
                    ballots,
                    warnings: warnings.clone(),
                };
                checkpoint.phase3 = Some(state.clone());
                checkpoint.requests_used = request_count.load(Ordering::SeqCst);
                save_checkpoint(&ckpt_path, &checkpoint)?;
                eprintln!(
                    "  {} voting phase: {} valid ballot(s)",
                    "✓".green(),
                    state.ballots.len()
                );
                state
            }
        };
        Some(state)
    };
    if let Some(state) = &phase3 {
        for warning in &state.warnings {
            if !warnings.contains(warning) {
                warnings.push(warning.clone());
            }
        }
    }

    // -- Phase 4: judge adjudication.
    let phase4 = match &phase3 {
        None => None,
        Some(p3) => {
            let state = match checkpoint.phase4.clone() {
                Some(state) => {
                    eprintln!("  {} adjudication reused from checkpoint", "·".dimmed());
                    state
                }
                None => {
                    // The profile's designated judge when set, otherwise the
                    // first council member (long-standing default). Selected
                    // from the unwrapped base providers and wrapped here, so
                    // an external designated judge consumes the same hard
                    // request budget as every council member.
                    let (judge_name, judge) = council::select_judge(&council, &base, &config)?;
                    let judge = budgeted_judge(
                        &base,
                        &judge_name,
                        judge,
                        Arc::clone(&request_count),
                        max_requests,
                    );
                    let started = std::time::Instant::now();
                    let timeout = run_deadline.turn_timeout(requested_timeout)?;
                    let prompt = judge_prompt(&phase1.findings, &p3.ballots);
                    let result = run_deadline
                        .wait(complete_with_timeout(
                            &judge,
                            &prompt,
                            Some(JUDGE_SYSTEM),
                            timeout,
                            &format!("judge `{judge_name}`"),
                        ))
                        .await?;
                    let latency_ms = started.elapsed().as_millis() as u64;
                    let state = match result {
                        Ok(text) => match parse_adjudications(&text).and_then(|adjudications| {
                            validate_adjudications(&adjudications, &phase1.findings)?;
                            Ok(adjudications)
                        }) {
                            Ok(adjudications) => Phase4State {
                                judge: ParticipantRecord {
                                    member: judge_name.clone(),
                                    transport: judge.transport().to_string(),
                                    status: "ok".to_string(),
                                    latency_ms,
                                    error: None,
                                },
                                adjudications,
                            },
                            Err(e) => anyhow::bail!(
                                "judge `{judge_name}` returned unparseable adjudications: {e} — \
                                 refusing to fabricate a verdict",
                            ),
                        },
                        Err(e) => anyhow::bail!("judge `{judge_name}` failed: {e}"),
                    };
                    checkpoint.phase4 = Some(state.clone());
                    checkpoint.requests_used = request_count.load(Ordering::SeqCst);
                    save_checkpoint(&ckpt_path, &checkpoint)?;
                    eprintln!(
                        "  {} adjudication: {} decision(s)",
                        "✓".green(),
                        state.adjudications.len()
                    );
                    state
                }
            };
            Some(state)
        }
    };

    // -- Assemble the receipt.
    let ballot_count = phase3.as_ref().map(|p| p.ballots.len()).unwrap_or(0);
    let findings: Vec<ReceiptFinding> = phase1
        .findings
        .iter()
        .map(|af| {
            let votes: Vec<ReceiptVote> = phase3
                .as_ref()
                .map(|p| {
                    p.ballots
                        .iter()
                        .flat_map(|b| {
                            b.votes.iter().filter(|v| v.finding_id == af.id).map(|v| ReceiptVote {
                                voter: b.voter.clone(),
                                vote: v.vote,
                                reason: v.reason.clone(),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let support = votes.iter().filter(|v| v.vote == Vote::Support).count();
            let oppose = votes.iter().filter(|v| v.vote == Vote::Oppose).count();
            let abstain = votes.iter().filter(|v| v.vote == Vote::Abstain).count();
            let cast = support + oppose + abstain;

            let adjudication = phase4
                .as_ref()
                .and_then(|p| p.adjudications.iter().find(|a| a.finding_id == af.id).cloned());
            let classification = match &adjudication {
                Some(a) if a.accepted => Some(classify(support, cast)),
                _ => None,
            };
            let dissent: Vec<ReceiptVote> = if classification.is_some() {
                votes.iter().filter(|v| v.vote != Vote::Support).cloned().collect()
            } else {
                Vec::new()
            };
            ReceiptFinding {
                id: af.id.clone(),
                title: af.finding.title.clone(),
                severity: af.finding.severity,
                file: af.finding.file.clone(),
                line: af.finding.line,
                evidence: af.finding.evidence.clone(),
                recommendation: af.finding.recommendation.clone(),
                source: af.source.clone(),
                member: af.member.clone(),
                votes,
                dissent,
                adjudication,
                classification,
            }
        })
        .collect();

    // Findings the judge never mentioned are honest non-decisions.
    if let Some(p4) = &phase4 {
        for f in &findings {
            if f.adjudication.is_none() {
                warnings.push(format!(
                    "adjudication: judge returned no decision for {}; recorded as not adjudicated",
                    f.id
                ));
            }
        }
        let _ = p4;
    }
    let _ = ballot_count;

    let member_entries: Vec<MemberEntry> = council
        .members
        .iter()
        .map(|m| MemberEntry {
            member: m.to_string(),
            utility: m.utility.as_str().to_string(),
            model: m.model.clone(),
            effort: m.effort.map(|e| e.as_str().to_string()),
            transport: m.utility.descriptor().transport.to_string(),
        })
        .collect();

    let receipt = ReviewReceipt {
        schema: 1,
        run_id: run_id.clone(),
        input: input.clone(),
        manifest: ManifestSection {
            profile: council.name.clone(),
            options: OptionsSection {
                quorum,
                deadline_secs,
                request_timeout_secs,
                max_concurrency,
                budget_usd,
                max_requests,
                options_hash: options_hash.clone(),
            },
            members: member_entries,
        },
        phases: PhasesSection {
            review: phase1.participants.clone(),
            voting: phase3.as_ref().map(|p| p.participants.clone()).unwrap_or_default(),
            adjudication: phase4.as_ref().map(|p| p.judge.clone()),
        },
        findings,
        warnings: warnings.clone(),
        budget_usd,
        budget_note: budget_usd
            .map(|_| "advisory only — transports expose no cost data; not enforced".to_string()),
        deadline_secs,
        request_timeout_secs,
        requests: RequestsSection { used: request_count.load(Ordering::SeqCst), max: max_requests },
        resumed,
    };

    if let Some(manifest) = &args.manifest {
        let json = serde_json::to_string_pretty(&receipt)?;
        write_atomic(manifest, &json)
            .with_context(|| format!("failed to write manifest {}", manifest.display()))?;
    }

    let rendered = match args.format.as_str() {
        "json" => serde_json::to_string_pretty(&receipt)?,
        _ => render_markdown(&receipt),
    };
    match &args.output {
        Some(path) => {
            std::fs::write(path, rendered)
                .with_context(|| format!("failed to write receipt {}", path.display()))?;
            eprintln!("{} Receipt written to {}", "✓".green(), path.display());
        }
        None => println!("{rendered}"),
    }

    for warning in &warnings {
        eprintln!("  {} {}", "⚠".yellow(), warning);
    }
    Ok(())
}

// Resolve config for tests without touching the filesystem.
#[allow(dead_code)]
fn default_council() -> Council {
    Config::default().resolve_profile(Some("deep")).expect("deep is a built-in profile")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use caucus_core::provider::{FanoutFailure, FanoutSuccess};
    use caucus_core::{ErrorKind, MockProvider};
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: ReviewArgs,
    }

    fn parse(args: &[&str]) -> std::result::Result<TestCli, clap::Error> {
        TestCli::try_parse_from(args)
    }

    // -- CLI conflicts and input selection ----------------------------------

    #[test]
    fn stdin_is_the_default_input() {
        let cli = parse(&["test"]).unwrap();
        assert!(cli.args.file.is_none());
        assert!(!cli.args.git_diff);
        assert!(!cli.args.staged);
        assert_eq!(cli.args.format, "markdown");
    }

    #[test]
    fn file_conflicts_with_git_inputs() {
        assert!(parse(&["test", "--file", "a.rs", "--git-diff"]).is_err());
        assert!(parse(&["test", "--file", "a.rs", "--staged"]).is_err());
    }

    #[test]
    fn git_diff_conflicts_with_staged() {
        assert!(parse(&["test", "--git-diff", "--staged"]).is_err());
    }

    #[test]
    fn profile_conflicts_with_auto() {
        assert!(parse(&["test", "--profile", "deep", "--auto"]).is_err());
    }

    #[test]
    fn format_is_validated() {
        assert!(parse(&["test", "--format", "yaml"]).is_err());
        assert!(parse(&["test", "--format", "json"]).is_ok());
    }

    #[test]
    fn budget_and_limits_parse() {
        let cli = parse(&[
            "test",
            "--quorum",
            "2",
            "--deadline-secs",
            "30",
            "--max-concurrency",
            "2",
            "--budget-usd",
            "1.5",
            "--max-requests",
            "7",
        ])
        .unwrap();
        assert_eq!(cli.args.quorum, Some(2));
        assert_eq!(cli.args.deadline_secs, Some(30));
        assert_eq!(cli.args.max_concurrency, Some(2));
        assert_eq!(cli.args.budget_usd, Some(1.5));
        assert_eq!(cli.args.max_requests, Some(7));
    }

    // -- FNV64 determinism ---------------------------------------------------

    #[test]
    fn fnv64_matches_reference_vectors() {
        // FNV-1a 64 offset basis for empty input.
        assert_eq!(fnv64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv64_hex(b""), "cbf29ce484222325");
        assert_eq!(fnv64_hex(b"a"), "af63dc4c8601ec8c");
        // Determinism: same input, same hash, every time.
        assert_eq!(fnv64(b"review me"), fnv64(b"review me"));
        assert_ne!(fnv64(b"review me"), fnv64(b"review me!"));
    }

    #[test]
    fn anon_source_ids_are_deterministic_and_anonymous() {
        let a = anon_source_id("claude:opus@xhigh");
        assert_eq!(a, anon_source_id("claude:opus@xhigh"));
        assert_ne!(a, anon_source_id("codex:default@xhigh"));
        assert!(a.starts_with("src-"));
        assert!(!a.contains("claude"), "source id must not leak the member: {a}");
    }

    // -- git argv construction ------------------------------------------------

    #[test]
    fn git_argv_is_explicit_and_shell_free() {
        assert_eq!(git_diff_argv(false), vec!["git", "diff"]);
        assert_eq!(git_diff_argv(true), vec!["git", "diff", "--staged"]);
    }

    #[test]
    fn truncated_git_diff_is_rejected_instead_of_reviewed_as_complete() {
        let output = ProcessOutput {
            stdout: "partial diff".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            latency_ms: 1,
            truncated: true,
        };
        let error = complete_git_diff(output, "git diff").unwrap_err().to_string();
        assert!(error.contains("incomplete diff"), "got: {error}");
    }

    // -- strict / fenced parsing ----------------------------------------------

    const FINDING_JSON: &str = r#"{"findings": [{"title": "SQL injection", "severity": "critical", "evidence": "query built with format!", "file": "src/db.rs", "line": 42, "recommendation": "use params"}]}"#;

    #[test]
    fn parses_strict_findings_document() {
        let findings = parse_findings(FINDING_JSON).unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Critical);
        assert_eq!(findings[0].line, Some(42));
    }

    #[test]
    fn parses_fenced_findings_and_bare_arrays() {
        let fenced =
            format!("Here are my findings:\n```json\n{FINDING_JSON}\n```\nHope this helps!");
        let findings = parse_findings(&fenced).unwrap();
        assert_eq!(findings.len(), 1);

        let bare = r#"[{"title": "t", "severity": "low", "evidence": "e", "file": null, "line": null, "recommendation": null}]"#;
        let findings = parse_findings(bare).unwrap();
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn prose_is_never_accepted_as_findings() {
        assert!(parse_findings("I found a bug on line 42, trust me.").is_err());
        assert!(parse_findings("").is_err());
        // Invalid severity is a schema error, not a finding.
        let bad = r#"{"findings": [{"title": "t", "severity": "catastrophic", "evidence": "e"}]}"#;
        assert!(parse_findings(bad).is_err());
        // Unknown fields violate the strict schema.
        let extra = r#"{"findings": [{"title": "t", "severity": "low", "evidence": "e", "confidence": 0.9}]}"#;
        assert!(parse_findings(extra).is_err());
    }

    #[test]
    fn parses_votes_strictly() {
        let votes = parse_votes(
            "```json\n{\"votes\": [{\"finding_id\": \"F1\", \"vote\": \"support\", \"reason\": \"evidence checks out\"}]}\n```",
        )
        .unwrap();
        assert_eq!(votes.len(), 1);
        assert_eq!(votes[0].vote, Vote::Support);
        assert!(parse_votes("I support F1.").is_err());
        let bad = r#"{"votes": [{"finding_id": "F1", "vote": "maybe"}]}"#;
        assert!(parse_votes(bad).is_err());
    }

    #[test]
    fn parses_adjudications_strictly() {
        let doc = r#"{"adjudications": [{"finding_id": "F1", "accepted": true, "reason": "supported by votes", "evidence": "line 42"}]}"#;
        let parsed = parse_adjudications(doc).unwrap();
        assert!(parsed[0].accepted);
        assert!(parse_adjudications("F1 is accepted.").is_err());
    }

    #[test]
    fn adjudications_must_cover_known_findings_exactly_once() {
        let finding = |id: &str| AttributedFinding {
            id: id.to_string(),
            source: "src-test".to_string(),
            member: "model".to_string(),
            finding: RawFinding {
                title: "title".to_string(),
                severity: Severity::High,
                evidence: "evidence".to_string(),
                file: None,
                line: None,
                recommendation: None,
            },
        };
        let decision = |id: &str| Adjudication {
            finding_id: id.to_string(),
            accepted: true,
            reason: "reason".to_string(),
            evidence: None,
        };
        let findings = vec![finding("F1"), finding("F2")];
        assert!(validate_adjudications(&[decision("F1"), decision("F2")], &findings).is_ok());
        assert!(validate_adjudications(&[decision("F1")], &findings).unwrap_err().contains("F2"));
        assert!(
            validate_adjudications(&[decision("F1"), decision("F1")], &findings)
                .unwrap_err()
                .contains("duplicate")
        );
        assert!(
            validate_adjudications(&[decision("F1"), decision("F3")], &findings)
                .unwrap_err()
                .contains("unknown")
        );
    }

    // -- vote classification ---------------------------------------------------

    #[test]
    fn classification_from_raw_votes() {
        assert_eq!(classify(3, 3), Classification::Unanimous);
        assert_eq!(classify(2, 3), Classification::Majority);
        assert_eq!(classify(1, 1), Classification::Unanimous);
        assert_eq!(classify(1, 2), Classification::Disputed);
        // No valid ballots: never fabricated as support.
        assert_eq!(classify(0, 0), Classification::Disputed);
        // Abstain-only is not unanimity.
        assert_eq!(classify(0, 2), Classification::Disputed);
    }

    #[test]
    fn classification_counts_every_valid_ballot() {
        // Regression: unanimous requires every valid ballot to support — one
        // support with the other ballots omitted or abstaining is not
        // unanimous.
        assert_eq!(classify(1, 3), Classification::Disputed);
        assert_ne!(classify(2, 3), Classification::Unanimous);
        assert_eq!(classify(3, 3), Classification::Unanimous);
        // Regression: majority requires support from more than half of all
        // valid ballots — one support plus two abstentions is not a majority,
        // and exactly half is not more than half.
        assert_eq!(classify(1, 3), Classification::Disputed);
        assert_eq!(classify(2, 4), Classification::Disputed);
        assert_eq!(classify(3, 4), Classification::Majority);
        assert_eq!(classify(2, 3), Classification::Majority);
        // Anything short of that is disputed.
        assert_eq!(classify(0, 3), Classification::Disputed);
    }

    // -- partial failure and quorum --------------------------------------------

    fn report(ok: usize, failed: usize, quorum: usize) -> FanoutReport {
        FanoutReport {
            successes: (0..ok)
                .map(|i| FanoutSuccess {
                    model: format!("m{i}"),
                    transport: Transport::Command,
                    content: "{}".into(),
                    latency_ms: 10,
                })
                .collect(),
            failures: (0..failed)
                .map(|i| FanoutFailure {
                    model: format!("bad{i}"),
                    transport: Transport::Api,
                    kind: ErrorKind::Timeout,
                    message: "boom".into(),
                    latency_ms: 5,
                })
                .collect(),
            quorum,
        }
    }

    #[test]
    fn quorum_is_enforced_per_phase() {
        assert!(enforce_quorum("review", 2, 2).is_ok());
        let err = enforce_quorum("review", 1, 2).unwrap_err();
        assert!(err.to_string().contains("quorum not met in review phase"));
    }

    #[test]
    fn review_quorum_counts_only_semantically_valid_responses() {
        // Regression: N unparseable responses must not satisfy quorum even
        // though every transport request succeeded.
        let mut r = report(2, 0, 2);
        r.successes[0].content = "I found bugs, trust me.".into();
        r.successes[1].content = "not json either".into();
        let (findings, valid, invalid, warnings) = collect_findings(&r);
        assert_eq!(valid, 0);
        assert_eq!(invalid.len(), 2);
        assert!(findings.is_empty());
        assert_eq!(warnings.len(), 2);
        assert!(enforce_quorum("review", valid, r.quorum).is_err());
        // Receipt errors are phase-correct: findings, not votes.
        assert!(invalid[0].error.contains("findings JSON"));
        assert_eq!(invalid[0].status, "parse-error");

        // A valid findings document with zero findings counts as a valid
        // review response.
        r.successes[0].content = "{\"findings\": []}".into();
        let (_, valid, invalid, _) = collect_findings(&r);
        assert_eq!(valid, 1);
        assert_eq!(invalid.len(), 1);
        assert!(enforce_quorum("review", valid, 1).is_ok());
    }

    #[test]
    fn incomplete_ballots_are_discarded_and_never_satisfy_quorum() {
        // Regression: ballots that miss findings, duplicate a vote, or vote
        // only on unknown IDs are not schema-complete and must not count.
        let known: std::collections::HashSet<&str> = ["F1", "F2"].into_iter().collect();
        let mut r = report(5, 0, 5);
        // Complete ballot: exactly one vote per known finding.
        r.successes[0].content =
            r#"{"votes": [{"finding_id": "F1", "vote": "support"}, {"finding_id": "F2", "vote": "abstain"}]}"#
                .into();
        // Missing a finding.
        r.successes[1].content = r#"{"votes": [{"finding_id": "F1", "vote": "support"}]}"#.into();
        // Duplicate vote for one finding.
        r.successes[2].content =
            r#"{"votes": [{"finding_id": "F1", "vote": "support"}, {"finding_id": "F1", "vote": "oppose"}, {"finding_id": "F2", "vote": "support"}]}"#
                .into();
        // Only unknown finding IDs.
        r.successes[3].content = r#"{"votes": [{"finding_id": "F9", "vote": "support"}]}"#.into();
        // Unparseable ballot.
        r.successes[4].content = "I support F1 and F2.".into();

        let (ballots, invalid, warnings) = collect_ballots(&r, &known);
        assert_eq!(ballots.len(), 1, "only the complete ballot counts");
        assert_eq!(ballots[0].votes.len(), 2);
        assert_eq!(invalid.len(), 4);
        assert!(enforce_quorum("voting", ballots.len(), 2).is_err());
        assert!(enforce_quorum("voting", ballots.len(), 1).is_ok());

        // Statuses and warnings are actionable and phase-correct.
        assert_eq!(invalid[0].status, "incomplete-ballot");
        assert!(invalid[0].error.contains("missing vote(s) for F2"));
        assert_eq!(invalid[1].status, "incomplete-ballot");
        assert!(invalid[1].error.contains("duplicate vote for `F1`"));
        assert_eq!(invalid[2].status, "incomplete-ballot");
        assert!(invalid[2].error.contains("unknown finding `F9`"));
        assert_eq!(invalid[3].status, "parse-error");
        assert!(invalid[3].error.contains("votes JSON"));
        assert!(warnings.iter().any(|w| w.contains("incomplete ballot")));
        assert!(!warnings.iter().any(|w| w.contains("findings JSON")));

        // Receipts carry the phase-correct status per participant.
        let records = participant_records(&r, &invalid);
        assert_eq!(records[0].status, "ok");
        assert_eq!(records[1].status, "incomplete-ballot");
        assert!(records[1].error.as_ref().unwrap().contains("missing vote(s) for F2"));
        assert_eq!(records[4].status, "parse-error");
        assert!(records[4].error.as_ref().unwrap().contains("votes JSON"));
    }

    #[test]
    fn participant_records_cover_success_parse_error_and_failure() {
        let mut r = report(2, 1, 1);
        r.successes[1].model = "m1".into();
        let invalid = vec![InvalidResponse {
            model: "m1".to_string(),
            status: "parse-error",
            error: "response was not valid votes JSON: nope".to_string(),
        }];
        let records = participant_records(&r, &invalid);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].status, "ok");
        assert!(records[0].error.is_none());
        assert_eq!(records[1].status, "parse-error");
        assert_eq!(records[1].error.as_deref(), Some("response was not valid votes JSON: nope"));
        assert_eq!(records[2].status, "failed");
        assert!(records[2].error.as_ref().unwrap().contains("timeout"));
    }

    // -- request budget ---------------------------------------------------------

    #[tokio::test]
    async fn max_requests_is_a_hard_cap() {
        let used = Arc::new(AtomicUsize::new(0));
        let provider = BudgetedProvider {
            inner: Arc::new(MockProvider::fixed("ok")),
            used: Arc::clone(&used),
            max: Some(2),
        };
        provider.complete("a", None).await.unwrap();
        provider.complete("b", None).await.unwrap();
        let err = provider.complete("c", None).await.unwrap_err();
        assert!(err.to_string().contains("--max-requests 2"));
        // The rejected request does not consume budget.
        assert_eq!(used.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn uncapped_budget_still_counts_everything() {
        // Regression: the counter is the receipt's request total, so every
        // admitted request increments it even with no cap configured.
        let used = Arc::new(AtomicUsize::new(0));
        let provider = BudgetedProvider {
            inner: Arc::new(MockProvider::fixed("ok")),
            used: Arc::clone(&used),
            max: None,
        };
        for _ in 0..5 {
            provider.complete("x", None).await.unwrap();
        }
        assert_eq!(used.load(Ordering::SeqCst), 5);
    }

    struct NeverProvider;

    #[async_trait::async_trait]
    impl LlmProvider for NeverProvider {
        async fn complete(&self, _prompt: &str, _system: Option<&str>) -> Result<String> {
            std::future::pending().await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn judge_request_timeout_is_enforced_without_a_run_deadline() {
        let error = complete_with_timeout(
            &NeverProvider,
            "judge",
            None,
            Duration::from_secs(3),
            "judge `slow`",
        )
        .await
        .unwrap_err();
        assert_eq!(caucus_core::ProviderError::classify(&error), caucus_core::ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn designated_judge_always_consumes_request_budget() {
        // Regression: a designated judge outside the council members is built
        // dedicated and must still consume the hard request budget.
        let member = |s: &str| s.parse().unwrap();
        let config = Config::default();
        let council = Council {
            name: "t".to_string(),
            description: None,
            members: vec![member("ollama:llama3.2:latest")],
            judge: Some(member("ollama:phi4")),
            strategy: "judge".to_string(),
            quorum: 1,
            deadline_secs: None,
            request_timeout_secs: None,
            budget_usd: None,
        };
        let base = council::build_council_provider(&council, &config).unwrap();

        // External designated judge: dedicated provider, budget-wrapped. With
        // the budget fully consumed by council phases, the adjudication
        // request fails honestly before any transport work is attempted.
        let (name, judge) = council::select_judge(&council, &base, &config).unwrap();
        assert!(matches!(judge, council::JudgeProvider::Owned(_)));
        let used = Arc::new(AtomicUsize::new(1));
        let judge = budgeted_judge(&base, &name, judge, Arc::clone(&used), Some(1));
        let err = judge.complete("adjudicate", None).await.unwrap_err();
        assert!(err.to_string().contains("--max-requests 1"));
        assert_eq!(used.load(Ordering::SeqCst), 1, "rejected request does not consume budget");

        // Member judge: borrowed from the base set and wrapped exactly once.
        let council = Council { judge: Some(member("ollama:llama3.2:latest")), ..council };
        let (name, judge) = council::select_judge(&council, &base, &config).unwrap();
        assert!(matches!(judge, council::JudgeProvider::Borrowed(_)));
        let used = Arc::new(AtomicUsize::new(0));
        let judge = budgeted_judge(&base, &name, judge, Arc::clone(&used), Some(0));
        let err = judge.complete("adjudicate", None).await.unwrap_err();
        assert!(err.to_string().contains("--max-requests 0"));
    }

    // -- checkpoint compatibility and resume ------------------------------------

    fn temp_checkpoint(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "caucus-review-test-{}-{}",
            std::process::id(),
            fnv64_hex(name.as_bytes())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("checkpoint.json")
    }

    #[test]
    fn checkpoint_roundtrips_and_validates_version() {
        let path = temp_checkpoint("roundtrip");
        let mut ckpt = Checkpoint::fresh("run1".into(), "in".into(), "opt".into());
        ckpt.phase1 =
            Some(Phase1State { participants: vec![], findings: vec![], warnings: vec![] });
        save_checkpoint(&path, &ckpt).unwrap();

        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.run_id, "run1");
        assert_eq!(loaded.input_hash, "in");
        assert!(loaded.phase1.is_some());
        assert!(loaded.phase3.is_none());

        // Unsupported versions are rejected.
        let mut raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        raw["version"] = serde_json::json!(99);
        std::fs::write(&path, raw.to_string()).unwrap();
        let err = load_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("version 99"));

        // Missing checkpoint is an honest error.
        let err = load_checkpoint(&path.with_file_name("nope.json")).unwrap_err();
        assert!(err.to_string().contains("no checkpoint"));
    }

    #[test]
    fn checkpoint_path_tracks_manifest() {
        assert_eq!(
            checkpoint_path(Some(Path::new("/tmp/m.json"))),
            PathBuf::from("/tmp/m.json.checkpoint.json")
        );
        assert_eq!(checkpoint_path(None), PathBuf::from(".caucus-review.checkpoint.json"));
    }

    #[test]
    fn checkpoint_writes_are_atomic_and_leave_no_temp_files() {
        // Writes go to a sibling temp file renamed over the target, so an
        // interruption can never leave a partially written resume file — and
        // normal operation leaves no temp files behind.
        let path = temp_checkpoint("atomic");
        let ckpt = Checkpoint::fresh("run1".into(), "in".into(), "opt".into());
        save_checkpoint(&path, &ckpt).unwrap();
        // Rewriting over an existing checkpoint replaces it in place.
        save_checkpoint(&path, &ckpt).unwrap();
        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.run_id, "run1");

        let dir = path.parent().unwrap();
        let leftovers: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "atomic writes left temp files: {leftovers:?}");
    }

    #[test]
    fn resume_rejects_mismatched_hashes() {
        // The exact comparison run() performs before reusing any phase.
        let ckpt = Checkpoint::fresh("run1".into(), "hash-a".into(), "opts-a".into());
        let matches =
            |input: &str, opts: &str| ckpt.input_hash == input && ckpt.options_hash == opts;
        assert!(matches("hash-a", "opts-a"));
        assert!(!matches("hash-b", "opts-a"));
        assert!(!matches("hash-a", "opts-b"));
    }

    #[test]
    fn checkpoint_requests_used_defaults_to_zero_and_roundtrips() {
        // Backward compatibility: checkpoints written before the field existed
        // still load, seeded at zero.
        let legacy = r#"{"version":1,"run_id":"r","input_hash":"i","options_hash":"o"}"#;
        let loaded: Checkpoint = serde_json::from_str(legacy).unwrap();
        assert_eq!(loaded.requests_used, 0);
        assert_eq!(Checkpoint::fresh("r".into(), "i".into(), "o".into()).requests_used, 0);

        // The cumulative total survives a save/load roundtrip.
        let path = temp_checkpoint("requests-used");
        let mut ckpt = Checkpoint::fresh("r".into(), "i".into(), "o".into());
        ckpt.requests_used = 7;
        save_checkpoint(&path, &ckpt).unwrap();
        assert_eq!(load_checkpoint(&path).unwrap().requests_used, 7);
    }

    #[tokio::test]
    async fn completed_resume_reuses_phases_and_reports_cumulative_total() {
        // Regression: an uncapped 3-member review makes 7 requests (3 review +
        // 3 vote + 1 judge). A resume of the completed run reuses every phase,
        // so the counter seeded from the checkpoint is the receipt's total.
        let used = Arc::new(AtomicUsize::new(0));
        let provider = BudgetedProvider {
            inner: Arc::new(MockProvider::fixed("ok")),
            used: Arc::clone(&used),
            max: None,
        };
        for _ in 0..7 {
            provider.complete("x", None).await.unwrap();
        }
        let mut ckpt = Checkpoint::fresh("r".into(), "i".into(), "o".into());
        ckpt.requests_used = used.load(Ordering::SeqCst);
        assert_eq!(ckpt.requests_used, 7);

        // Resume: run() seeds the counter from the checkpoint; with all phases
        // reused no new requests are admitted, so the total stays cumulative.
        let resumed = AtomicUsize::new(ckpt.requests_used);
        assert_eq!(resumed.load(Ordering::SeqCst), 7);
    }

    #[tokio::test]
    async fn capped_resume_seeds_counter_and_judge_stays_denied() {
        // Regression: with --max-requests 6 a 3-member run exhausts the budget
        // in the voting phase (3 review + 3 vote) and checkpoints
        // requests_used = 6. On resume the counter starts at 6, so the judge
        // remains denied and the denied request is not counted.
        let ckpt = {
            let mut c = Checkpoint::fresh("r".into(), "i".into(), "o".into());
            c.requests_used = 6;
            c
        };
        let used = Arc::new(AtomicUsize::new(ckpt.requests_used));
        let judge = BudgetedProvider {
            inner: Arc::new(MockProvider::fixed("ok")),
            used: Arc::clone(&used),
            max: Some(6),
        };
        let err = judge.complete("adjudicate", None).await.unwrap_err();
        assert!(err.to_string().contains("--max-requests 6"));
        assert_eq!(used.load(Ordering::SeqCst), 6, "denied request is not counted");
    }

    // -- receipt rendering -------------------------------------------------------

    fn sample_receipt() -> ReviewReceipt {
        ReviewReceipt {
            schema: 1,
            run_id: "abc123".into(),
            input: InputMeta {
                source: "file".into(),
                detail: "src/main.rs".into(),
                bytes: 128,
                hash: "deadbeef".into(),
            },
            manifest: ManifestSection {
                profile: "deep".into(),
                options: OptionsSection {
                    quorum: 3,
                    deadline_secs: Some(600),
                    request_timeout_secs: Some(240),
                    max_concurrency: 4,
                    budget_usd: Some(1.0),
                    max_requests: Some(20),
                    options_hash: "optshash".into(),
                },
                members: vec![MemberEntry {
                    member: "kimi:kimi-code/k3@high".into(),
                    utility: "kimi".into(),
                    model: "kimi-code/k3".into(),
                    effort: Some("high".into()),
                    transport: "command".into(),
                }],
            },
            phases: PhasesSection {
                review: vec![ParticipantRecord {
                    member: "kimi:kimi-code/k3@high".into(),
                    transport: "command".into(),
                    status: "ok".into(),
                    latency_ms: 120,
                    error: None,
                }],
                voting: vec![],
                adjudication: None,
            },
            findings: vec![ReceiptFinding {
                id: "F1".into(),
                title: "Unchecked unwrap".into(),
                severity: Severity::High,
                file: Some("src/main.rs".into()),
                line: Some(10),
                evidence: "value.unwrap()".into(),
                recommendation: Some("handle the error".into()),
                source: "src-01234567".into(),
                member: "kimi:kimi-code/k3@high".into(),
                votes: vec![
                    ReceiptVote {
                        voter: "claude:opus@xhigh".into(),
                        vote: Vote::Support,
                        reason: Some("evidence is exact".into()),
                    },
                    ReceiptVote {
                        voter: "codex:default@xhigh".into(),
                        vote: Vote::Oppose,
                        reason: Some("infallible here".into()),
                    },
                ],
                dissent: vec![ReceiptVote {
                    voter: "codex:default@xhigh".into(),
                    vote: Vote::Oppose,
                    reason: Some("infallible here".into()),
                }],
                adjudication: Some(Adjudication {
                    finding_id: "F1".into(),
                    accepted: true,
                    reason: "majority support with cited evidence".into(),
                    evidence: Some("value.unwrap()".into()),
                }),
                classification: Some(Classification::Disputed),
            }],
            warnings: vec!["budget_usd is advisory".into()],
            budget_usd: Some(1.0),
            budget_note: Some("advisory only".into()),
            deadline_secs: Some(600),
            request_timeout_secs: Some(240),
            requests: RequestsSection { used: 11, max: Some(20) },
            resumed: false,
        }
    }

    #[test]
    fn markdown_receipt_is_provenance_rich() {
        let md = render_markdown(&sample_receipt());
        for needle in [
            "abc123",
            "deadbeef",
            "src/main.rs:10",
            "kimi:kimi-code/k3@high",
            "disputed",
            "majority support with cited evidence",
            "infallible here",
            "budget_usd is advisory",
            "11/20 used",
        ] {
            assert!(md.contains(needle), "markdown receipt missing `{needle}`:\n{md}");
        }
    }

    #[test]
    fn json_receipt_serializes_all_sections() {
        let json = serde_json::to_string_pretty(&sample_receipt()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["run_id"], "abc123");
        assert_eq!(value["input"]["hash"], "deadbeef");
        assert_eq!(value["manifest"]["options"]["quorum"], 3);
        assert_eq!(value["manifest"]["members"][0]["effort"], "high");
        assert_eq!(value["phases"]["review"][0]["latency_ms"], 120);
        assert_eq!(value["findings"][0]["classification"], "disputed");
        assert_eq!(value["findings"][0]["votes"][1]["vote"], "oppose");
        assert_eq!(value["findings"][0]["adjudication"]["accepted"], true);
        assert_eq!(value["requests"]["used"], 11);
        assert_eq!(value["warnings"][0], "budget_usd is advisory");
    }

    // -- deep profile sanity ------------------------------------------------------

    #[test]
    fn deep_profile_has_exact_members() {
        let council = default_council();
        let members: Vec<String> = council.members.iter().map(|m| m.to_string()).collect();
        assert_eq!(
            members,
            vec![
                "claude:opus@xhigh",
                "claude:claude-fable-5@xhigh",
                "codex:default@xhigh",
                "opencode:zai-coding-plan/glm-5.2@xhigh",
                "kimi:kimi-code/k3@high",
            ]
        );
    }
}
