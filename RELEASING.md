# Releasing ITMO_FS

## Version source of truth

The package version lives in `ITMO_FS/VERSION`.

Both `pyproject.toml` and `ITMO_FS/__about__.py` read the version from that
file, so the release process should never edit version strings in multiple
places.

## Release flow

1. Update `ITMO_FS/VERSION` with the new version.
2. Sync a local release-capable environment:
   - `uv sync --group dev`
3. Verify packaging and tests locally:
   - `uv run --no-sync pytest -q`
   - `uv run --no-sync python -m build`
4. Update any release-facing metadata that should match the new public release:
   - `CITATION.cff`
   - `meta.yml`
   - release notes / GitHub release text
5. Commit the release changes.
6. Create and push a Git tag in the form `vX.Y.Z`.
7. GitHub Actions publishes the built package to PyPI from that tag.

## Notes

- The publish workflow uses PyPI trusted publishing via GitHub Actions.
- Configure the PyPI project to trust this repository before the first release.
- For development snapshots, use a PEP 440 version such as `0.3.5.dev1`.
- `uv sync --no-default-groups --group test` gives a local test environment without docs/data tools.
- `uv sync --no-default-groups --group docs` installs only documentation tooling.
- `uv sync --only-group release` installs only build/release tooling.
