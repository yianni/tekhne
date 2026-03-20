# Contributing

Thanks for contributing to `tekhne`.

`tekhne` is intentionally small and learning-first. Favor clarity, correctness, and explicit code over clever abstractions.

## Philosophy

Prefer:
- small, focused pull requests
- explicit and readable code
- changes that fit `tekhne`
- improvements that keep the core easy to understand

## Branch names

Use short, descriptive branch names with a simple prefix when possible.

Examples:
- `feat/add-dataset-shuffling`
- `fix/install-sbt-in-ci`
- `docs/refine-contributing-guidelines`
- `test/add-shape-validation-cases`

## Commit messages

Use short, descriptive commit messages in a conventional style when possible.

Common prefixes:
- `feat`
- `fix`
- `docs`
- `test`
- `refactor`
- `ci`
- `chore`

Examples:
- `feat: add dataset shuffling to training`
- `fix: install sbt in CI`
- `docs: add README usage example`
- `test: add numeric gradient coverage for small network`

Prefer one clear change per commit.

## Pull requests

Keep pull requests small and focused.

A good pull request should:
- explain what changed
- explain why it belongs in `tekhne`
- include verification
- call out anything intentionally left out

If a change is large, split it into smaller issues or PRs.

## Local checks

Before opening a PR, run the checks that make sense for your change.

```bash
sbt test
sbt scalafmtAll
sbt scalafixAll
sbt "project core" clean coverage test coverageReport
```

For demo-related changes:

```bash
sbt "demo/runMain tekhne.demo.runTekhne"
sbt "demo/runMain tekhne.demo.runAndGateDemo"
```
