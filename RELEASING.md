# Releasing ITMO_FS

## Version source of truth

The package version lives in `ITMO_FS/VERSION`.

Both `pyproject.toml` and `ITMO_FS/__about__.py` read the version from that
file, so the release process should never edit version strings in multiple
places.

## Release flow

1. Update `ITMO_FS/VERSION` with the new version.
2. Verify packaging and tests locally:
   - `.venv/bin/python -m build`
   - `.venv/bin/pytest -q tests`
3. Update any release-facing metadata that should match the new public release:
   - `CITATION.cff`
   - `meta.yml`
   - release notes / GitHub release text
4. Commit the release changes.
5. Create and push a Git tag in the form `vX.Y.Z`.
6. GitHub Actions publishes the built package to PyPI from that tag.

## Notes

- The publish workflow uses PyPI trusted publishing via GitHub Actions.
- Configure the PyPI project to trust this repository before the first release.
- For development snapshots, use a PEP 440 version such as `0.3.5.dev1`.
