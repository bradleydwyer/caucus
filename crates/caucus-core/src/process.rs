//! Safe subprocess runtime for CLI-backed providers.
//!
//! All command execution goes through [`run_argv`]: an explicit argv vector
//! (never a shell), controlled stdin/env/cwd, `kill_on_drop`, a wall-clock
//! timeout, and capped stdout/stderr capture. Errors are normalized into
//! [`ProviderError`] so callers can classify failures without inspecting
//! OS-specific error text.
//!
//! Children start from a scrubbed environment (`env_clear`) with only a
//! documented safe baseline restored (see [`SAFE_ENV_ALLOWLIST`]). A caller
//! may additionally inherit exact, adapter-specific variables via
//! [`ProcessSpec::inherit_env`] or provide explicit overrides via
//! [`ProcessSpec::env`]. This keeps ambient secrets out by default while
//! allowing a known CLI adapter to receive only the auth/config/proxy state it
//! actually needs.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::error::{ErrorKind, ProviderError};

/// A command to run: program plus explicit arguments. Never a shell string.
#[derive(Debug, Clone)]
pub struct ProcessSpec {
    /// Executable name or path (looked up on PATH if not absolute).
    pub program: String,
    /// Arguments, in order. Each is passed verbatim — no shell expansion.
    pub args: Vec<String>,
    /// Bytes written to the child's stdin (stdin is closed immediately after).
    pub stdin: Option<String>,
    /// Explicit environment overrides applied after the scrubbed-allowlist
    /// environment is built (see [`run_argv`]). Values are never logged.
    pub env: Vec<(String, String)>,
    /// Exact parent environment names to inherit in addition to the safe
    /// baseline. Intended for adapter-specific auth/config/proxy variables.
    pub inherit_env: Vec<String>,
    /// Working directory for the child (defaults to the current directory).
    pub cwd: Option<PathBuf>,
}

impl ProcessSpec {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            stdin: None,
            env: Vec::new(),
            inherit_env: Vec::new(),
            cwd: None,
        }
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn stdin(mut self, input: impl Into<String>) -> Self {
        self.stdin = Some(input.into());
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Inherit one exact variable from the parent environment if it exists.
    pub fn inherit_env(mut self, key: impl Into<String>) -> Self {
        self.inherit_env.push(key.into());
        self
    }

    pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    /// The full argv vector (program first), for logging and tests.
    pub fn argv(&self) -> Vec<String> {
        std::iter::once(self.program.clone()).chain(self.args.iter().cloned()).collect()
    }
}

/// Limits applied to a child process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProcessLimits {
    /// Wall-clock timeout; the child is killed when it expires.
    pub timeout: Duration,
    /// Maximum stdout bytes retained; excess is discarded and flagged.
    pub max_stdout_bytes: usize,
    /// Maximum stderr bytes retained; excess is discarded and flagged.
    pub max_stderr_bytes: usize,
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            max_stdout_bytes: 1024 * 1024,
            max_stderr_bytes: 256 * 1024,
        }
    }
}

impl ProcessLimits {
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Captured result of a completed child process.
#[derive(Debug, Clone)]
pub struct ProcessOutput {
    pub stdout: String,
    pub stderr: String,
    /// Exit code; `None` if the process was terminated by a signal.
    pub exit_code: Option<i32>,
    /// Wall-clock time from spawn to exit.
    pub latency_ms: u64,
    /// True when stdout and/or stderr exceeded the configured cap.
    pub truncated: bool,
}

impl ProcessOutput {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// Exact-name environment variables restored after `env_clear`.
///
/// Everything here is non-secret and needed for a CLI to behave sanely:
/// program lookup, basic identity, locale/terminal settings, temp dirs, and
/// the minimal Windows system variables. Anything not listed (or not matching
/// [`SAFE_ENV_PREFIXES`]) is scrubbed from the child environment unless the
/// caller explicitly inherits it with [`ProcessSpec::inherit_env`] or
/// supplies it via [`ProcessSpec::env`].
pub const SAFE_ENV_ALLOWLIST: &[&str] = &[
    // Program lookup and basic session identity.
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    // Terminal and locale behaviour.
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    // Temporary directories.
    "TMPDIR",
    "TEMP",
    "TMP",
    // Minimal Windows system variables (no-ops elsewhere).
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
];

/// Environment variable name prefixes restored after `env_clear`.
///
/// `XDG_*` covers the freedesktop base-directory spec (config/data/cache/
/// runtime dirs and their search paths); none of these hold credentials.
pub const SAFE_ENV_PREFIXES: &[&str] = &["XDG_"];

/// Run `spec` to completion under `limits`.
///
/// The child is spawned directly from the argv vector (never via a shell)
/// with `kill_on_drop`. Its environment is scrubbed with `env_clear`, then
/// only [`SAFE_ENV_ALLOWLIST`] / [`SAFE_ENV_PREFIXES`] entries inherited from
/// this process are restored, then exact `spec.inherit_env` names, and finally
/// `spec.env` overrides. Stdin is written and closed, and stdout/stderr are
/// drained concurrently (capped) by reader tasks that are always awaited
/// before this function returns. A non-zero exit is an error carrying the
/// tail of stderr; a missing executable maps to [`ErrorKind::Unavailable`];
/// a timeout explicitly kills and reaps the child and maps to
/// [`ErrorKind::Timeout`].
pub async fn run_argv(spec: &ProcessSpec, limits: &ProcessLimits) -> Result<ProcessOutput> {
    let started = Instant::now();
    let mut cmd = tokio::process::Command::new(&spec.program);
    cmd.args(&spec.args)
        .kill_on_drop(true)
        .stdin(if spec.stdin.is_some() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    cmd.process_group(0);
    // Scrub the inherited environment, restore only the safe non-secret
    // allowlist, then apply explicit caller overrides.
    cmd.env_clear();
    for (key, value) in std::env::vars_os() {
        let listed = key.to_str().is_some_and(|k| {
            SAFE_ENV_ALLOWLIST.contains(&k)
                || SAFE_ENV_PREFIXES.iter().any(|p| k.starts_with(p))
                || spec.inherit_env.iter().any(|name| name == k)
        });
        if listed {
            cmd.env(key, value);
        }
    }
    for (key, value) in &spec.env {
        cmd.env(key, value);
    }
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }

    let mut child = cmd.spawn().map_err(|e| {
        let kind = if e.kind() == std::io::ErrorKind::NotFound {
            ErrorKind::Unavailable
        } else {
            ErrorKind::Other
        };
        ProviderError::new(kind, format!("failed to spawn `{}`: {e}", spec.program))
    })?;
    let process_id = child.id();
    #[cfg(unix)]
    let mut process_group = ProcessGroupGuard::new(process_id);

    if let Some(input) = &spec.stdin {
        let mut stdin = child.stdin.take().expect("stdin piped");
        let input = input.clone();
        // Write on a task so a large prompt can't deadlock against a child
        // that stops reading; ignore EPIPE (child exited early).
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(input.as_bytes()).await;
            let _ = stdin.shutdown().await;
        });
    }

    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");
    let out_limit = limits.max_stdout_bytes;
    let err_limit = limits.max_stderr_bytes;
    let stdout_capture = Arc::new(Mutex::new(Capture::default()));
    let stderr_capture = Arc::new(Mutex::new(Capture::default()));
    let out_shared = Arc::clone(&stdout_capture);
    let err_shared = Arc::clone(&stderr_capture);
    let mut out_task =
        tokio::spawn(async move { read_capped(&mut stdout, out_limit, out_shared).await });
    let mut err_task =
        tokio::spawn(async move { read_capped(&mut stderr, err_limit, err_shared).await });

    let status = match tokio::time::timeout(limits.timeout, child.wait()).await {
        Ok(status) => status,
        Err(_) => {
            terminate_process_tree(&mut child, process_id).await;
            #[cfg(unix)]
            process_group.disarm();
            let _ = tokio::time::timeout(PIPE_DRAIN_TIMEOUT, async {
                let _ = tokio::join!(&mut out_task, &mut err_task);
            })
            .await;
            out_task.abort();
            err_task.abort();
            return Err(ProviderError::timeout(format!(
                "`{}` exceeded {}s timeout",
                spec.program,
                limits.timeout.as_secs()
            ))
            .into());
        }
    };
    let status =
        status.map_err(|e| ProviderError::new(ErrorKind::Other, format!("wait failed: {e}")))?;
    let mut out = None;
    let mut err = None;
    let mut pipes_held_open = false;
    if tokio::time::timeout(
        PIPE_DRAIN_TIMEOUT,
        drain_readers(&mut out_task, &mut err_task, &mut out, &mut err),
    )
    .await
    .is_err()
    {
        pipes_held_open = true;
        // The direct child exited but a descendant kept an inherited output
        // pipe open. Kill that descendant group, then finish only the reader
        // tasks that did not complete during the first drain attempt.
        #[cfg(unix)]
        process_group.kill();
        if tokio::time::timeout(
            PIPE_DRAIN_TIMEOUT,
            drain_readers(&mut out_task, &mut err_task, &mut out, &mut err),
        )
        .await
        .is_err()
        {
            // A process outside the child's group can still hold a leaked
            // pipe (notably the macOS pipe()/fcntl() close-on-exec race).
            // Preserve bytes captured so far and flag truncation instead of
            // turning a successful direct child into a false timeout.
            if out.is_none() {
                out_task.abort();
                let _ = (&mut out_task).await;
            }
            if err.is_none() {
                err_task.abort();
                let _ = (&mut err_task).await;
            }
        }
    }
    #[cfg(unix)]
    process_group.disarm();

    validate_reader_result(out, pipes_held_open, "stdout")?;
    validate_reader_result(err, pipes_held_open, "stderr")?;
    let (stdout, out_truncated) = snapshot_capture(&stdout_capture);
    let (stderr, err_truncated) = snapshot_capture(&stderr_capture);

    let output = ProcessOutput {
        stdout,
        stderr,
        exit_code: status.code(),
        latency_ms: started.elapsed().as_millis() as u64,
        truncated: out_truncated || err_truncated || pipes_held_open,
    };

    if !output.success() {
        let detail = output.stderr.lines().last().unwrap_or("").trim();
        return Err(ProviderError::new(
            ErrorKind::Other,
            format!(
                "`{}` exited with code {:?}{}",
                spec.program,
                output.exit_code,
                if detail.is_empty() { String::new() } else { format!(": {detail}") }
            ),
        )
        .into());
    }

    Ok(output)
}

/// Maximum time allowed for descendants to close inherited output pipes
/// after their direct child exits or is killed.
#[cfg(not(test))]
const PIPE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(test)]
const PIPE_DRAIN_TIMEOUT: Duration = Duration::from_millis(100);

type ReaderResult = std::result::Result<std::io::Result<()>, tokio::task::JoinError>;

async fn drain_readers(
    out_task: &mut tokio::task::JoinHandle<std::io::Result<()>>,
    err_task: &mut tokio::task::JoinHandle<std::io::Result<()>>,
    out: &mut Option<ReaderResult>,
    err: &mut Option<ReaderResult>,
) {
    while out.is_none() || err.is_none() {
        tokio::select! {
            result = &mut *out_task, if out.is_none() => *out = Some(result),
            result = &mut *err_task, if err.is_none() => *err = Some(result),
        }
    }
}

fn validate_reader_result(
    result: Option<ReaderResult>,
    allowed_incomplete: bool,
    stream: &str,
) -> Result<()> {
    match result {
        Some(joined) => joined
            .map_err(|error| {
                ProviderError::new(ErrorKind::Other, format!("{stream} task failed: {error}"))
            })?
            .map_err(|error| {
                ProviderError::new(ErrorKind::Other, format!("{stream} read failed: {error}"))
            }),
        None if allowed_incomplete => Ok(()),
        None => {
            Err(ProviderError::new(ErrorKind::Other, format!("{stream} reader did not complete")))
        }
    }
    .map_err(Into::into)
}

async fn terminate_process_tree(child: &mut tokio::process::Child, process_id: Option<u32>) {
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        // Every command is placed in its own process group before spawn.
        unsafe {
            libc::kill(-(process_id as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill().await;
    let _ = tokio::time::timeout(PIPE_DRAIN_TIMEOUT, child.wait()).await;
}

/// Cancellation safety for Unix: dropping `run_argv` (for example when an
/// outer whole-run deadline expires) terminates the command's process group,
/// including descendants. Disarmed immediately after a normal child exit.
#[cfg(unix)]
struct ProcessGroupGuard {
    pgid: Option<i32>,
}

#[cfg(unix)]
impl ProcessGroupGuard {
    fn new(process_id: Option<u32>) -> Self {
        Self { pgid: process_id.map(|id| id as i32) }
    }

    fn disarm(&mut self) {
        self.pgid = None;
    }

    fn kill(&mut self) {
        if let Some(pgid) = self.pgid.take() {
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

#[derive(Default)]
struct Capture {
    bytes: Vec<u8>,
    truncated: bool,
}

fn snapshot_capture(capture: &Arc<Mutex<Capture>>) -> (String, bool) {
    let capture = capture.lock().expect("capture mutex poisoned");
    (String::from_utf8_lossy(&capture.bytes).into_owned(), capture.truncated)
}

/// Read to EOF, keeping at most `limit` bytes in shared capture state. The
/// shared buffer lets the caller preserve bytes if a leaked pipe forces the
/// reader task to be aborted after the direct child has already exited.
async fn read_capped<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
    capture: Arc<Mutex<Capture>>,
) -> std::io::Result<()> {
    use tokio::io::AsyncReadExt;
    let mut chunk = [0u8; 8192];
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let mut state = capture.lock().expect("capture mutex poisoned");
        let room = limit.saturating_sub(state.bytes.len());
        if room > 0 {
            state.bytes.extend_from_slice(&chunk[..n.min(room)]);
        }
        if n > room {
            state.truncated = true;
        }
    }
    Ok(())
}

/// Locate an executable on PATH without invoking a shell. Returns the first
/// matching absolute path, or `None` when the binary is not found.
pub fn find_on_path(program: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(program);
    if candidate.is_absolute() {
        return candidate.exists().then_some(candidate);
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(program);
        if candidate.is_file() && is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata().map(|m| m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &std::path::Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ProcessLimits {
        ProcessLimits::default().with_timeout(Duration::from_secs(10))
    }

    #[tokio::test]
    async fn run_argv_captures_stdout() {
        let spec = ProcessSpec::new("/bin/echo").arg("hello");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert!(out.success());
        assert_eq!(out.stdout.trim(), "hello");
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn run_argv_writes_stdin() {
        let spec = ProcessSpec::new("/bin/cat").stdin("piped in");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert_eq!(out.stdout, "piped in");
    }

    #[tokio::test]
    async fn run_argv_nonzero_exit_is_error_with_stderr() {
        let spec = ProcessSpec::new("/bin/sh").args(["-c", "echo boom >&2; exit 3"]);
        let err = run_argv(&spec, &limits()).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains('3') && msg.contains("boom"), "got: {msg}");
    }

    #[tokio::test]
    async fn run_argv_missing_binary_is_unavailable() {
        let spec = ProcessSpec::new("definitely-not-a-real-binary-xyz");
        let err = run_argv(&spec, &limits()).await.unwrap_err();
        assert_eq!(ProviderError::classify(&err), ErrorKind::Unavailable);
    }

    #[tokio::test]
    async fn run_argv_timeout_is_classified() {
        let spec = ProcessSpec::new("/bin/sleep").arg("30");
        let tight = limits().with_timeout(Duration::from_millis(50));
        let err = run_argv(&spec, &tight).await.unwrap_err();
        assert_eq!(ProviderError::classify(&err), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn run_argv_caps_output_and_flags_truncation() {
        let spec = ProcessSpec::new("/bin/cat").stdin("x".repeat(10_000));
        let tight = ProcessLimits { max_stdout_bytes: 100, ..limits() };
        let out = run_argv(&spec, &tight).await.unwrap();
        assert_eq!(out.stdout.len(), 100);
        assert!(out.truncated);
    }

    #[tokio::test]
    async fn run_argv_applies_env_overrides() {
        // Explicit spec.env overrides reach the child even for names that are
        // not on the inherited allowlist.
        let spec = ProcessSpec::new("/usr/bin/env").env("CAUCUS_TEST_VAR", "override-works");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert!(out.stdout.lines().any(|l| l == "CAUCUS_TEST_VAR=override-works"));
    }

    #[tokio::test]
    async fn run_argv_inherits_only_explicit_adapter_variable() {
        let inherited = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets manifest dir");
        let spec = ProcessSpec::new("/usr/bin/env").inherit_env("CARGO_MANIFEST_DIR");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert!(out.stdout.lines().any(|line| line == format!("CARGO_MANIFEST_DIR={inherited}")));
        assert!(!out.stdout.lines().any(|line| line.starts_with("CARGO_PKG_NAME=")));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_descendant_process_group() {
        let marker = std::env::temp_dir().join(format!(
            "caucus-descendant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let script = format!("sleep 30 & echo $! > '{}'; wait", marker.display());
        let spec = ProcessSpec::new("/bin/sh").args(["-c", &script]);
        let err =
            run_argv(&spec, &ProcessLimits::default().with_timeout(Duration::from_millis(250)))
                .await
                .unwrap_err();
        assert_eq!(ProviderError::classify(&err), ErrorKind::Timeout);
        let pid: i32 = std::fs::read_to_string(&marker).unwrap().trim().parse().unwrap();
        let _ = std::fs::remove_file(&marker);

        let mut gone = false;
        for _ in 0..20 {
            let alive = unsafe { libc::kill(pid, 0) } == 0;
            if !alive {
                gone = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(gone, "descendant process {pid} survived timeout cleanup");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn successful_child_keeps_output_when_descendant_holds_pipe_open() {
        let spec = ProcessSpec::new("/bin/sh").args(["-c", "sleep 30 & echo complete"]);
        let output = run_argv(&spec, &limits()).await.unwrap();
        assert_eq!(output.stdout.trim(), "complete");
        assert!(output.success());
        assert!(output.truncated, "forced pipe cleanup must be visible to callers");
    }

    #[tokio::test]
    async fn run_argv_scrubs_non_allowlisted_env() {
        assert!(std::env::var_os("CARGO_MANIFEST_DIR").is_some());
        let spec = ProcessSpec::new("/usr/bin/env");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert!(
            !out.stdout.lines().any(|l| l.starts_with("CARGO_MANIFEST_DIR=")),
            "non-allowlisted variable leaked into child: {}",
            out.stdout
        );
    }

    #[tokio::test]
    async fn run_argv_restores_allowlisted_env() {
        // Allowlisted variables (e.g. PATH) survive env_clear.
        let spec = ProcessSpec::new("/usr/bin/env");
        let out = run_argv(&spec, &limits()).await.unwrap();
        assert!(out.stdout.lines().any(|l| l.starts_with("PATH=")), "got: {}", out.stdout);
    }

    #[test]
    fn argv_is_program_first() {
        let spec = ProcessSpec::new("prog").args(["a", "b"]);
        assert_eq!(spec.argv(), vec!["prog", "a", "b"]);
    }
}
