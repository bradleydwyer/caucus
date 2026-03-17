---
name: caucus
description: "Multi-LLM consensus engine. Queries multiple AI models and synthesizes a single answer using strategies like voting, judge synthesis, and multi-round debate. Use when the user wants to query multiple LLMs, compare model outputs, get a consensus answer, or needs a synthesized response from several AI models."
allowed-tools:
  - Bash(caucus:*)
  - Bash(cat:*)
  - Bash(mktemp:*)
  - Read
  - Write
  - Grep
  - Glob
user-invocable: true
argument-hint: "[prompt to send to multiple models]"
metadata:
  author: bradleydwyer
  version: "0.1.0"
  status: experimental
---

# Caucus -- Multi-LLM Consensus Engine

Queries multiple LLM models on a single prompt, then produces one consensus result. Strategies range from simple majority vote to multi-round debate with a judge synthesizing the final answer.

## When to Use This Skill

- User wants to ask multiple LLMs the same question
- User wants a consensus or synthesized answer across models
- User wants to compare how different models respond
- User needs higher confidence on a tricky question
- User asks to "debate" or "vote" on something across models

## Prerequisites

API keys as env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`. Only keys for models you want to query are required.

If `caucus --help` fails, tell the user to install it before proceeding.

## Workflow

### 1. Determine Intent

| User Says | Action |
|---|---|
| "ask multiple models", "get consensus" | Standard consensus query |
| "compare models", "how do different models answer" | Strategy comparison via `compare` |
| "debate this", "have models argue about" | Debate strategy (`-s debate`) |
| specific model names mentioned | Select with `-m` |

### 2. Run the Query

Always use `-f json` for parseable output.

```bash
caucus "prompt" -f json                                          # all models, judge strategy
caucus "prompt" -m gpt-5.2,claude-opus-4-6 -f json              # specific models
caucus "prompt" -s debate -f json                                # debate strategy
caucus compare "prompt" --strategies majority-vote,judge -f json # compare strategies
caucus "prompt" -v -f json                                       # verbose (individual responses)
caucus "prompt" -f supreme-court                                 # dissent/concurrence format
```

**Strategies:** `majority-vote`, `weighted-vote`, `judge` (default), `debate`, `debate-then-vote`

**Output formats:** `plain` (default), `json`, `supreme-court`, `detailed`

### 3. Present Results

Parse the JSON and present:
- **The consensus answer** -- lead with this, prominently
- **Strategy used** and model agreement/divergence
- **Individual model positions** -- summarize briefly if there was disagreement
- **Confidence level** -- how strong was the consensus?

For `compare` runs, present each strategy's result and which produced the strongest consensus.

Use `-v` to re-run if the user wants full reasoning or individual model responses.

## Tips

- **Judge is the default for good reason.** It synthesizes rather than counting votes -- better for nuanced questions.
- **Debate works best for controversial or ambiguous topics** where models should challenge each other.
- **Majority vote is fastest** but loses nuance. Use for factual questions with clear answers.
- **Use `compare`** when unsure which strategy fits.
- **Fewer models = faster.** Use `-m` to pick 2-3 when speed matters.
