# Contributing

This project is a public alpha. Bug reports, model-adapter fixes, reproducibility findings, and benchmark feedback are welcome.

## Setup

Use Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

Install only the optional model runtime needed for a model, for example `.[gliner2]` or `.[laya]`.

## Changes

- Keep changes small and reuse the existing worker, registry, CLI, and result formats.
- Add a regression test for behavior that plausibly breaks again.
- Run the test suite and build distributions before opening a pull request.
- Document user-visible CLI or result-format changes in `README.md` and `CHANGELOG.md`.
- Pin external model and GitHub Action revisions immutably.

Do not commit API keys, credential files, model weights, caches, raw/private datasets, or generated prediction files. Use synthetic or aggregate examples in reports and tests.

## Reports

Use GitHub Issues for reproducible bugs and benchmark/model feedback. Use the private process in `SECURITY.md` for security-sensitive reports.
