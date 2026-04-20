# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `ITMO_FS/` with subpackages for `filters`, `wrappers`, `embedded`, `ensembles`, `hybrid`, shared `base`, and `utils`.
- Tests sit in `tests/` with dataset fixtures in `tests/datasets/`; mirror the package layout when adding new coverage.
- Docs and assets are under `docs/` (Sphinx), published to Read the Docs; keep diagrams and logos in `docs/logos/`.
- Distribution artifacts are built into `dist/`; avoid committing local build outputs.

## Build, Test, and Development Commands
- Install for local dev: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- Run unit tests: `pytest` (use `pytest -n auto` to parallelize).
- Lint quickly: `ruff check ITMO_FS tests`.
- Optional typing pass: `mypy ITMO_FS` (when adding or refactoring typed code).
- Build wheel/sdist before release: `python -m build` (requires `build` in your env).

## Coding Style & Naming Conventions
- Target Python ≥3.8 with 4-space indents and PEP 8 defaults; keep lines ≤120 chars.
- Use snake_case for functions/variables, CapWords for classes, and prefix private helpers with `_`.
- Prefer type hints for new/modified APIs; keep docstrings concise and actionable.
- Keep public interfaces sklearn-friendly (`fit`, `transform`, `fit_transform`), and ensure new measures/cutting rules stay composable.

## Testing Guidelines
- Framework: pytest; place new suites near related modules (e.g., `tests/univariate_filters_test.py` patterns).
- Name test functions `test_*` and group fixtures/helpers in the same file when small or in a dedicated `conftest.py` if reused.
- For algorithms, include deterministic seeds and small synthetic datasets to avoid flaky results.
- Aim to extend coverage for new parameters and failure modes (shape/dtype checks, edge cases on sparse input).

## Commit & Pull Request Guidelines
- Commits: short, imperative subject lines (e.g., “add reliefF docstring”, “fix coverage”); keep logical changes grouped.
- PRs: describe intent, major changes, and verification steps (tests/linters run); link issues when applicable.
- Include examples or benchmarks when altering algorithm behavior, and update docs/tutorials if user-facing behavior shifts.
- Keep diffs focused; prefer follow-up PRs over unrelated drive-by changes.
