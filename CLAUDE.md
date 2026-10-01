# Conventions for Claude Code

- Work one phase at a time; don't start the next until the done-when check passes.
- Every connector and pipeline step gets a pytest test against small fixtures in `tests/fixtures/` — no network calls in tests (mock or record responses).
- Type hints everywhere; ruff clean.
- Never write into `raw/`; originals are immutable.
- Every function that creates or transforms data writes a provenance record.
- No hardcoded paths, credentials, or API keys — config and environment variables only.
- Manual fallbacks are features, not afterthoughts: document them in the CLI help.
- Before adding a dependency, check its license; flag GPL/AGPL for review.
