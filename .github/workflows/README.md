# GitHub Actions Workflows

## Publish to PyPI

Automatically publishes the package to PyPI when a GitHub Release is published.

**Trigger**: 
- When a release is published on GitHub
- Manual trigger via `workflow_dispatch`

**Requirements**:
- PyPI trusted publishing must be configured (see [PyPI documentation](https://docs.pypi.org/trusted-publishers/))
- The workflow uses `pypa/gh-action-pypi-publish` which supports trusted publishing

## Update Version in README

Automatically updates the version number in the README citation section when a new tag is pushed.

**Trigger**: 
- When a tag matching pattern `v*` is pushed (e.g., `v0.1.3`)

**What it does**:
- Extracts version from tag (removes `v` prefix)
- Updates `version = {...}` in the README citation section
- Commits and pushes the change

**Note**: This runs before the PyPI publish workflow, ensuring the README is up-to-date.

