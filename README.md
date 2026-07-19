# caucus

Multi-LLM consensus engine. Query multiple models, get one answer.

caucus takes responses from several LLMs and produces a single consensus result. Several strategies are available: voting, judge synthesis, and multi-round debate. Rust core with a CLI, HTTP API, MCP server, and Python bindings.

## Install

```bash
brew install bradleydwyer/tap/caucus
```

Or from source:

```bash
git clone https://github.com/bradleydwyer/caucus
cd caucus
cargo install --path crates/caucus-cli
```

### Python library (optional)

Requires [maturin](https://github.com/PyO3/maturin) to compile the Rust code into a Python module:

```bash
pip install maturin
maturin develop --release
```

Then: `from caucus import consensus, Candidate`

## Quick start

Already signed in to one of the supported CLIs? You do not need API keys.

```bash
# See what is installed and ready
caucus doctor

# Build a council from ready CLIs and local models
caucus ask --auto "What causes inflation?"

# Or use the built-in frontier profile
caucus ask --profile deep "What causes inflation?"

# Show the members and the final reasoning
caucus ask --profile deep --verbose --format detailed "What causes inflation?"
```

Direct API models still work:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

caucus "What causes inflation?" -m gpt-5.2,claude-opus-4-6
```

For the direct API form, `ask` is optional. caucus detects configured API
models, uses `judge`, and prints the answer.

## Strategies

| Strategy | LLM needed? | Description |
|----------|-------------|-------------|
| `majority-vote` | No | Groups responses by similarity, picks the largest group |
| `weighted-vote` | No | Same as majority but weighted by confidence or model reputation |
| `judge` | Yes | A separate LLM evaluates all responses and synthesizes the best one (default) |
| `debate` | Yes | Multi-round debate where positions are refined until convergence |
| `debate-then-vote` | Yes | Debate rounds followed by majority vote |

With a single model, caucus skips consensus and returns the response directly.

## Output formats

| Format | Use case | Example |
|--------|----------|---------|
| `plain` | Just the consensus text (default) | [plain.md](examples/plain.md) |
| `json` | Full result with metadata | [json.md](examples/json.md) |
| `supreme-court` | Majority opinion + concurrences + dissents | [supreme-court.md](examples/supreme-court.md) |
| `detailed` | Full transcript with all candidates | [detailed.md](examples/detailed.md) |

See also: [verbose output](examples/verbose.md), [debate with supreme-court format](examples/debate-supreme-court.md)

To watch a debate as it happens, add `--live`. Initial positions are shown
before the first round, then each model's full round response is written to
stderr as soon as it is ready. The final result remains on stdout. Use
`--format plain` to avoid replaying the transcript after the live output.

```bash
caucus debate "The gold price will increase over the next 12 months" \
  --rounds 3 --live --format plain
```

## CLI commands

```bash
caucus "prompt"
caucus ask "prompt" --strategy debate --format supreme-court
caucus ask "prompt" --profile deep          # council profile (exact members)
caucus ask "prompt" --auto                  # ready subscription CLIs + local models
caucus review --git-diff                    # adjudicated multi-model code review
caucus compare "prompt" --strategies majority-vote,judge
caucus debate "prompt" --rounds 3 --live
caucus bench tests.jsonl -o results.json
caucus profiles                             # list built-in and user profiles
caucus doctor                               # adapter readiness + config health
caucus serve --port 8080
caucus serve --mcp
```

## Councils and profiles

A **council** is a named panel of exact member specs (`utility:model@effort`).
`ask` and `review` accept `--profile NAME` (defaulting to the configured
`default_profile`, then the built-in `deep`) or `--auto`, which assembles a
zero-key council from locally discovered CLI adapters and local servers. No
API keys required, and every exclusion is reported with its reason.

The built-in `deep` profile (quorum 3, whole-run deadline 600s, judge strategy) has
exactly these members:

```
claude:opus@xhigh
claude:claude-fable-5@xhigh
codex:default@xhigh
opencode:zai-coding-plan/glm-5.2@xhigh
kimi:kimi-code/k3@high
```

Adapter compatibility labels, shown by `caucus doctor`:

| Label | Adapters | Meaning |
|-------|----------|---------|
| `stable` | claude, codex, ollama, lmstudio | Supported, expected to work |
| `experimental` | kimi, opencode, gemini, grok, acp | Works, but flags/pins may shift |

```bash
caucus doctor          # readiness, config validity, profile health (or --json)
caucus profiles        # all profiles with resolved members (or a name, or --json)
```

Kimi effort is set for each child process with
`KIMI_MODEL_THINKING_EFFORT`; caucus does not edit Kimi's config. Grok Build
accepts `low`, `medium`, and `high` only, so its strongest profile spec is
`grok:grok-4.5@high`.

## Adjudicated review

`caucus review` is a fixed four-phase review pipeline (not a generic DAG):

1. **Review:** each member independently reviews the input and returns
   findings under a strict JSON schema (direct or fenced JSON only; prose is
   never accepted as findings).
2. **Anonymize:** findings get deterministic anonymous source IDs.
3. **Vote:** blind peer support/oppose/abstain votes with evidence-linked
   reasons over all anonymized findings.
4. **Adjudicate:** a council judge accepts or rejects each finding and cites
   the evidence and raw votes.

Accepted findings are classified `unanimous`, `majority`, or `disputed` from
the raw votes. Partial peer failures are warnings; review and voting require
quorum. Adjudication uses the profile's judge. Bad model output never becomes
a made-up vote or verdict.

```bash
caucus review --file src/main.rs              # exactly one input:
caucus review --git-diff                      #   --file, --git-diff,
caucus review --staged                        #   --staged, or stdin (default)
git diff | caucus review
caucus review --staged --profile deep --format json --output receipt.json
# Start a checkpointed run
caucus review --git-diff --manifest run.json

# Resume it after an interruption
caucus review --git-diff --manifest run.json --resume
```

The decision receipt is Markdown or JSON. It includes run and input hashes,
member details, per-phase status, raw votes, and adjudication. Warnings plus
deadline and request-count data are included too. See
[examples/review-receipt.md](examples/review-receipt.md).

Each completed review stage is checkpointed to a small versioned JSON file next
to `--manifest` (or `.caucus-review.checkpoint.json`). `--resume` checks the
input and options before reusing completed work.

`--budget-usd` is **advisory only**: current transports expose no cost data,
so it is recorded with a warning but not enforced. `--max-requests` is a hard
cap across all phases, and it is cumulative across resumed invocations. The
request count persists in the checkpoint, so a cap applies to the whole run,
not to each `caucus review` invocation.

Git input runs as explicit argv (`git diff [--staged]`). No shell involved.

## HTTP API

The server binds to `127.0.0.1` by default and has no authentication. Do not
expose it to a LAN or the internet. HTTP requests cannot run installed command
or ACP adapters. Use the normal CLI commands for installed utilities. ACP is
not implemented yet.

```bash
caucus serve --port 8080

curl -X POST http://localhost:8080/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": ["response 1", "response 2", "response 3"],
    "strategy": "majority_vote",
    "format": "json"
  }'
```

`/v1/consensus` also accepts `judge_model` for judge/debate strategies.
`/v1/pipeline` runs configured pipeline steps over HTTP providers. `caucus
serve --mcp` currently exposes majority and weighted consensus over supplied
answers.

## Rust library

```rust
use caucus_core::{consensus, Candidate};

let candidates = vec![
    Candidate::new("The answer is 42").with_model("gpt-5.2"),
    Candidate::new("The answer is 42").with_model("claude-opus-4-6"),
    Candidate::new("The answer is 7").with_model("gemini-3.1-pro-preview"),
];

let result = consensus(&candidates, "majority_vote", None).await?;
println!("{}", result.content);         // "The answer is 42"
println!("{:.0}%", result.agreement_score * 100.0); // "67%"
```

## Python

```python
from caucus import consensus, Candidate

candidates = [
    Candidate(content="The answer is 42", model="gpt-5.2"),
    Candidate(content="The answer is 42", model="claude-opus-4-6"),
    Candidate(content="The answer is 7", model="gemini-3.1-pro-preview"),
]

result = consensus(candidates, strategy="majority_vote")
print(result.content)          # "The answer is 42"
print(result.agreement_score)  # 0.67
```

## Claude Code Skill

caucus includes a [skill](SKILL.md) for Claude Code. Install it with
[equip](https://github.com/bradleydwyer/equip):

```bash
equip install bradleydwyer/caucus
```

This lets Claude Code use caucus directly when you ask it to query multiple models or get a consensus answer.

## Configuration

API keys are read from environment variables or a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...
```

The CLI auto-loads `.env` from the current directory. You can also pass `--env path/to/.env`.

**Credential handling:** caucus does not scrape credentials. Discovery checks
PATH and local server ports. The Gemini adapter checks that `GEMINI_API_KEY`
exists, but never reads or prints its value.

Profile adapters use each CLI's existing login. `grok:` uses Grok Build;
legacy `--models grok-*` uses the xAI API. Kimi effort is passed through the
non-secret `KIMI_MODEL_THINKING_EFFORT` child environment variable.

### Config file

Profile and adapter config lives in TOML. Discovery order:
`--config PATH`, then `./caucus.toml`, then
`$XDG_CONFIG_HOME/caucus/config.toml`, then `~/.config/caucus/config.toml`.
See [examples/config.toml](examples/config.toml) and
[examples/caucus.toml](examples/caucus.toml).

```toml
default_profile = "deep"          # falls back to built-in `deep` when unset

[profiles.frontier]               # user profiles shadow built-ins
description = "..."
strategy = "judge"                # strategy used by `ask`
quorum = 3                        # default: member count
deadline_secs = 600               # whole-run wall-clock limit
request_timeout_secs = 240        # maximum for each provider request
budget_usd = 5.0                  # advisory (see review docs)
judge = "claude:opus@max"         # optional designated judge (default: first member)
members = [
  "claude:opus@xhigh",
  "codex:default@xhigh",
  "kimi:kimi-code/k3@high",
  "grok:grok-4.5@high",
]

[adapters.claude]                 # per-adapter overrides
binary_path = "/usr/local/bin/claude"
timeout_secs = 900
max_stdout_bytes = 2097152
max_stderr_bytes = 262144
env = { NO_COLOR = "1" }         # passed only to this adapter process

[adapters.ollama]
model = "llama3.2:latest"         # pin used by --auto when discovery finds none
```

Member specs are `utility:model@effort`; the model id is passed through
verbatim. Effort support: claude `[low, medium, high, xhigh, max]`, codex
`[minimal, low, medium, high, xhigh]`, kimi `[low, high, max]`, opencode all
six, grok `[low, medium, high]`. No effort: ollama / lmstudio / gemini / acp.

**Legacy profile fields:** the previous schema's `models`, `timeout_seconds`,
and `deadline_seconds` keys are still accepted. They are migrated in memory
only. Your file is never edited, and every migration prints a
`config warning:` telling you exactly what changed: `models` becomes
`members`, `timeout_seconds` becomes `request_timeout_secs`, and
`deadline_seconds` becomes the whole-run `deadline_secs`. When both legacy
fields are present, both limits are preserved.
Legacy member strings work too: a pin-less entry like `codex@xhigh` means
`codex:default@xhigh`, and the `glm` utility alias resolves to `opencode`.

## License

MIT

## More Tools

**Naming & Availability**

- [available](https://github.com/bradleydwyer/available): AI-powered project name finder (uses parked, staked & published)
- [parked](https://github.com/bradleydwyer/parked): Domain availability checker (DNS → WHOIS → RDAP)
- [staked](https://github.com/bradleydwyer/staked): Package registry name checker (npm, PyPI, crates.io + 19 more)
- [published](https://github.com/bradleydwyer/published): App store name checker (App Store & Google Play)

**AI Tooling**

- [sloppy](https://github.com/bradleydwyer/sloppy): AI prose/slop detector
- [nanaban](https://github.com/bradleydwyer/nanaban): Gemini image generation CLI
- [equip](https://github.com/bradleydwyer/equip): Cross-agent skill manager
