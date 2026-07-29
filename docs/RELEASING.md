# Releasing ERINYS Memory

ERINYS Memory publishes to PyPI through GitHub Actions Trusted Publishing.
No long-lived PyPI token is stored in GitHub.

## One-time PyPI configuration

Configure the `erinys-memory` project with this GitHub publisher:

| Field | Value |
|---|---|
| Owner | `GhostyAI-HA` |
| Repository | `ERINYS-mem` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The workflow identity is defined in `.github/workflows/release.yml`.

## Normal release

1. Update the package version and `CHANGELOG.md`.
2. Merge the tested release change into `main`.
3. Publish a non-prerelease GitHub Release whose tag is `v<package-version>`.
4. Confirm the `Publish to PyPI` workflow succeeds.
5. Verify the version and files on the public PyPI project.

The workflow builds from the release tag, verifies that the tag matches
`pyproject.toml`, validates both distributions with Twine, and publishes through
a short-lived OIDC credential.

## Publishing an existing release

Use the manual trigger only when the GitHub Release already exists:

```bash
gh workflow run release.yml \
  --repo GhostyAI-HA/ERINYS-mem \
  --ref main \
  -f tag=v0.6.0
```

The same tag/version check runs before publishing. PyPI rejects replacement of
an existing distribution, so use a new version for any changed artifact.
