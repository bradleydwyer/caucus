# conroute

Multi-LLM consensus engine — composable strategies for aggregating and synthesizing outputs from multiple LLMs.

## About

conroute takes responses from multiple LLMs and produces a single consensus result. It provides several strategies (voting, judge synthesis, multi-round debate) as composable primitives — not tied to any agent framework.

Rust core with a CLI, HTTP API, MCP server, and Python bindings.

## Installation

### From source

```bash
git clone https://github.com/conroute/conroute
cd conroute
cargo build --release
```

The binary is at `./target/release/conroute`.

### Python (via maturin)

```bash
pip install maturin
maturin develop --release
```

## Quick start

```bash
# Set API keys (or put them in .env)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...

# One-shot consensus across three models
conroute ask "What causes inflation?" \
  --models gpt-5.2,claude-opus-4-6,gemini-3.1-pro-preview \
  --strategy majority-vote \
  --format supreme-court
```

## Strategies

| Strategy | LLM needed? | Description |
|----------|-------------|-------------|
| `majority-vote` | No | Groups responses by similarity, picks the largest group |
| `weighted-vote` | No | Same as majority but weighted by confidence or model reputation |
| `judge` | Yes | A separate LLM evaluates all responses and synthesizes the best one |
| `debate` | Yes | Multi-round debate where positions are refined until convergence |
| `debate-then-vote` | Yes | Debate rounds followed by majority vote |

## Output formats

| Format | Use case |
|--------|----------|
| `plain` | Just the consensus text |
| `json` | Full result with metadata, for programmatic use |
| `supreme-court` | Majority opinion + concurrences + dissents + vote summary |
| `detailed` | Full transcript with all candidates and process info |

## CLI commands

```bash
# One-shot consensus
conroute ask "prompt" --models gpt-5.2,claude-opus-4-6 --strategy judge

# Compare strategies side-by-side
conroute compare "prompt" --models gpt-5.2,claude-opus-4-6 --strategies majority-vote,judge

# Multi-round debate
conroute debate "prompt" --models gpt-5.2,claude-opus-4-6 --rounds 3

# Batch evaluation
conroute bench tests.jsonl --models gpt-5.2 --strategies majority-vote,judge -o results.json

# HTTP API server
conroute serve --port 8080

# MCP server (stdio)
conroute serve --mcp
```

Use `conroute <command> --help` to see all options.

## HTTP API

```bash
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
use conroute_core::{consensus, Candidate};

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
from conroute import consensus, Candidate

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
```

The CLI auto-loads `.env` from the current directory, or you can specify a path with `--env path/to/.env`.

## Development

```bash
cargo test          # Run all tests
cargo clippy        # Lint
cargo run -p conroute-core --example basic_consensus
```

## License

MIT OR Apache-2.0
