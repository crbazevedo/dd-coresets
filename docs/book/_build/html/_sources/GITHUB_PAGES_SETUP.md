# GitHub Pages Setup Guide

## Problem: 404 Error After Deployment

If you're getting a 404 error at `https://crbazevedo.github.io/dd-coresets/` even though the workflow ran successfully, GitHub Pages is likely not enabled in the repository settings.

## Solution: Enable GitHub Pages

### Option 1: Via GitHub Web Interface (Recommended)

1. Go to your repository: https://github.com/crbazevedo/dd-coresets
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar, under "Code and automation")
4. Under **Source**, select:
   - **Deploy from a branch**
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
5. Click **Save**

The site should be available at `https://crbazevedo.github.io/dd-coresets/` within a few minutes.

### Option 2: Via GitHub CLI

```bash
# Check current Pages configuration
gh api repos/crbazevedo/dd-coresets/pages

# Enable Pages (if not already enabled)
gh api repos/crbazevedo/dd-coresets/pages \
  --method POST \
  -f source[type]=branch \
  -f source[branch]=gh-pages \
  -f source[path]=/
```

### Option 3: Verify gh-pages Branch Exists

The workflow `peaceiris/actions-gh-pages@v3` automatically creates the `gh-pages` branch on first deployment. Verify it exists:

```bash
git fetch origin gh-pages
git branch -r | grep gh-pages
```

## Troubleshooting

### Workflow Runs But Pages Not Available

1. **Check workflow logs**: Ensure the "Deploy to GitHub Pages" step completed successfully
2. **Wait a few minutes**: GitHub Pages can take 1-5 minutes to propagate
3. **Check repository settings**: Ensure Pages is enabled (see Option 1 above)
4. **Verify branch exists**: The `gh-pages` branch should exist after first deployment

### Still Getting 404?

1. Clear browser cache
2. Try incognito/private mode
3. Check if the URL is correct: `https://crbazevedo.github.io/dd-coresets/`
4. Verify the workflow created files in `gh-pages` branch:
   ```bash
   git checkout gh-pages
   ls -la  # Should see HTML files
   ```

## Current Workflow Configuration

The workflow (`.github/workflows/deploy-docs.yml`) is configured to:
- Build Jupyter Book from `docs/book/`
- Deploy to `gh-pages` branch
- Publish directory: `docs/book/_build/html`

## Expected Behavior

After enabling GitHub Pages:
1. Workflow runs on push to `main` (when `docs/book/` changes)
2. Builds Jupyter Book
3. Deploys to `gh-pages` branch
4. GitHub Pages serves from `gh-pages` branch
5. Site available at: `https://crbazevedo.github.io/dd-coresets/`

## Verification

Once enabled, you can verify:
- Repository Settings → Pages shows "Your site is live at..."
- The `gh-pages` branch exists with HTML files
- The URL loads correctly

