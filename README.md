# caucus

> **Work in progress.** This is an active experiment exploring whether multi-LLM consensus actually produces better results than a single good model. APIs, output formats, and strategies may change without notice.

Multi-LLM consensus engine — composable strategies for aggregating and synthesizing outputs from multiple LLMs.

## About

caucus takes responses from multiple LLMs and produces a single consensus result. It provides several strategies (voting, judge synthesis, multi-round debate) as composable primitives — not tied to any agent framework.

Rust core with a CLI, HTTP API, MCP server, and Python bindings.

## Installation

### Homebrew (macOS)

```bash
brew install bradleydwyer/tap/caucus
```

### From source

```bash
git clone https://github.com/bradleydwyer/caucus
cd caucus
cargo install --path crates/caucus-cli
```

### Python library (optional)

Only needed if you want to use caucus from Python. Requires [maturin](https://github.com/PyO3/maturin) to compile the Rust code into a native Python module.

```bash
pip install maturin
maturin develop --release  # builds and installs into your current Python env
```

Then: `from caucus import consensus, Candidate`

## Quick start

```bash
# Set API keys (or put them in .env)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...
export XAI_API_KEY=xai-...

# Just ask — queries all configured models, synthesizes the best answer
caucus "What causes inflation?"

# Pick your models
caucus "What causes inflation?" -m gpt-5.2,claude-opus-4-6,gemini-3.1-pro-preview,grok-4-1-fast-reasoning

# See what's happening under the hood
caucus "What causes inflation?" -v

# Override strategy and format
caucus "What causes inflation?" -s debate -f supreme-court
```

No subcommand required. caucus auto-detects configured models, uses `judge` strategy by default, and prints just the answer.

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
| `json` | Full result with metadata, for programmatic use | [json.md](examples/json.md) |
| `supreme-court` | Majority opinion + concurrences + dissents + vote summary | [supreme-court.md](examples/supreme-court.md) |
| `detailed` | Full transcript with all candidates and process info | [detailed.md](examples/detailed.md) |

See also: [verbose output](examples/verbose.md), [debate strategy with supreme-court format](examples/debate-supreme-court.md)

## CLI commands

```bash
# Default — just works
caucus "prompt"

# Explicit subcommands for power users
caucus ask "prompt" --strategy debate --format supreme-court
caucus compare "prompt" --strategies majority-vote,judge
caucus debate "prompt" --rounds 3
caucus bench tests.jsonl -o results.json
caucus serve --port 8080
caucus serve --mcp
```

## HTTP API

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

## Configuration

API keys are read from environment variables. You can set them directly or use a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...
```

The CLI auto-loads `.env` from the current directory, or you can specify a path with `--env path/to/.env`.

## Development

```bash
cargo test          # Run all tests
cargo clippy        # Lint
cargo run -p caucus-core --example basic_consensus
```

## License

MIT
