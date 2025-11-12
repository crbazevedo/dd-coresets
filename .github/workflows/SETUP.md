# Setting Up Automated PyPI Publishing

This repository uses GitHub Actions to automatically publish to PyPI when a release is created.

## Prerequisites

1. **PyPI Account**: You need a PyPI account with the project `dd-coresets` registered
2. **Trusted Publishing**: Configure PyPI trusted publishing (recommended) or use API tokens

## Option 1: Trusted Publishing (Recommended)

Trusted publishing is the most secure method and doesn't require storing API tokens.

### Steps:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Navigate to **"Publishing"** → **"Add a new pending publisher"**
3. Fill in:
   - **PyPI project name**: `dd-coresets`
   - **Owner**: `crbazevedo` (your GitHub username)
   - **Repository name**: `dd-coresets`
   - **Workflow filename**: `publish-pypi.yml`
   - **Environment name**: (leave empty for default)
4. Click **"Add"**

The workflow will automatically use trusted publishing when you create a release.

## Option 2: API Token (Alternative)

If you prefer using API tokens:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create an API token with scope: `dd-coresets`
3. Add it as a GitHub secret:
   - Go to repository Settings → Secrets and variables → Actions
   - Add secret: `PYPI_API_TOKEN` with your token value
4. Update `.github/workflows/publish-pypi.yml` to use the token instead of trusted publishing

## Testing

To test the workflow:

1. Create a test release:
   ```bash
   git tag -a v0.1.4-test -m "Test release"
   git push origin v0.1.4-test
   ```

2. Or trigger manually via GitHub Actions UI (workflow_dispatch)

## Workflow Behavior

- **`update-version.yml`**: Runs when a tag `v*` is pushed, updates README citation version
- **`publish-pypi.yml`**: Runs when a GitHub Release is published, builds and publishes to PyPI

## Notes

- The badge in README uses `badge.fury.io` which automatically fetches the latest PyPI version
- The citation version is automatically updated by the `update-version.yml` workflow
- Always ensure `pyproject.toml` version matches the tag version before creating a release

