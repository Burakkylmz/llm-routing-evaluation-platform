# Privacy and Execution Safety

**Status:** Approved policy; sanitizer implementation pending.

- Use synthetic development fixtures; no real banking data in the initial MVP.
- No secrets or unsanitized identifiers in logs, traces, reports or Git.
- Run offline by default; later live runs require explicit opt-in and credentials.
- Sanitize before third-party calls and persistence; block when policy cannot be met.
- Redaction does **not** guarantee irreversible anonymization.
- Keep golden answer/reference material out of router and embedding inputs.

See [decisions.md](decisions.md), sections 8 and 9 for adopted controls.
