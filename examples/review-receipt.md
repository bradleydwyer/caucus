<!-- Example `caucus review` Markdown receipt, produced offline by a fake-adapter
     council (`binary_path` adapter overrides pointing at a canned-response script)
     so the structure is visible without network access. The run is capped with
     `--max-requests 10` so the request provenance line is visible. -->

# Caucus Review Receipt

- Run: `bc19fe72c7d7a634`
- Input: file (`input.rs`), 52 bytes, hash `f00dcecfd07f8450`
- Profile: `smoke` (quorum 2, concurrency 4)
- Requests: 7/10 used

## Members

- `claude:fake@high` — utility `claude`, model `fake`, effort `high`, transport `command`
- `codex:default@high` — utility `codex`, model `default`, effort `high`, transport `command`
- `kimi:fake@high` — utility `kimi`, model `fake`, effort `high`, transport `command`

## Phase participants

### review

- `claude:fake@high`: ok (3 ms)
- `codex:default@high`: ok (5 ms)
- `kimi:fake@high`: ok (3 ms)

### voting

- `claude:fake@high`: ok (6 ms)
- `codex:default@high`: ok (7 ms)
- `kimi:fake@high`: ok (6 ms)

### adjudication

- `claude:fake@high`: ok (5 ms)

## Findings

### F1 [low] — accepted — unanimous

fake finding

- Evidence: fake adapter evidence
- Source: `src-9c6821a8` (member `claude:fake@high`)
- Adjudication: supported by the votes
- Votes:
  - `claude:fake@high`: support — evidence checks out
  - `codex:default@high`: support — evidence checks out
  - `kimi:fake@high`: support — evidence checks out
### F2 [low] — accepted — unanimous

fake finding

- Evidence: fake adapter evidence
- Source: `src-be469c5e` (member `codex:default@high`)
- Adjudication: supported by the votes
- Votes:
  - `claude:fake@high`: support — evidence checks out
  - `codex:default@high`: support — evidence checks out
  - `kimi:fake@high`: support — evidence checks out

### F3 [low] — accepted — unanimous

fake finding

- Evidence: fake adapter evidence
- Source: `src-6d46ef53` (member `kimi:fake@high`)
- Adjudication: supported by the votes
- Votes:
  - `claude:fake@high`: support — evidence checks out
  - `codex:default@high`: support — evidence checks out
  - `kimi:fake@high`: support — evidence checks out
