# Contributing

**Current phase:** Member 01 contracts, orchestration and offline demo are implemented. Members 02–04 must review and implement owned adapters; repository-wide quality checks are not yet verified.

- Use short-lived feature branches and pull requests; never push directly to protected `main`.
- Require at least one independent reviewer, plus affected owners for public contract changes.
- Keep implementation typed, tested, and consistent with SOLID and Ports & Adapters.
- Document changes to contracts with versioning and compatibility rationale.
- Never commit API keys, personal data, real customer inputs or unsafe generated traces.
- CI must remain offline. Live provider/judge calls require explicit opt-in outside CI.
- Require Ruff, mypy and pytest checks before release. Existing scaffold CI must be updated by Member 04.
- Offline Member 01 reproduction instructions are in [member01-implementation.md](member01-implementation.md).

See [decisions.md](decisions.md) for the engineering baseline.
