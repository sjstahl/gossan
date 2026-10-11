# Conventions for Claude Code

- Work one phase at a time; don't start the next until the done-when check passes.
- Every connector and pipeline step gets a pytest test against small fixtures in `tests/fixtures/` — no network calls in tests (mock or record responses).
- Type hints everywhere; ruff clean.
- Never write into `raw/`; originals are immutable.
- Every function that creates or transforms data writes a provenance record.
- No hardcoded paths, credentials, or API keys — config and environment variables only.
- Manual fallbacks are features, not afterthoughts: document them in the CLI help.
- Before adding a dependency, check its license; flag GPL/AGPL for review.
- A default that rests on a statistical claim needs a held-out or effort-controlled check first. Put the numbers and the area they come from in the setting's comment, so the next person knows how far to trust it.
- Outputs that depend on data under a restrictive licence must inherit it (`derived_license_from`); never claim a looser licence than the inputs carry.
- Downloads go through `core/rawstore.store_raw` (never overwrite, never keep identical copies).
