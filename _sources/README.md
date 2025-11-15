# Jupyter Book Documentation

This directory contains the source files for the dd-coresets documentation, built with Jupyter Book.

## Structure

- `_config.yml` - Jupyter Book configuration
- `_toc.yml` - Table of contents
- `intro.md` - Introduction page
- `installation.md` - Installation guide
- `quickstart.md` - Quick start guide
- `tutorials/` - Tutorial notebooks
- `concepts/` - Conceptual/theoretical content
- `guides/` - Practical guides
- `api/` - API reference
- `use_cases/` - Use case examples

## Building Locally

```bash
# Install jupyter-book
pip install jupyter-book

# Build the book
cd docs/book
jupyter-book build .

# View locally
# Open docs/book/_build/html/index.html
```

## Deployment

The documentation is automatically deployed to GitHub Pages via `.github/workflows/deploy-docs.yml` when changes are pushed to `main`.

## Adding Content

1. Add markdown files to appropriate directories
2. Update `_toc.yml` to include new pages
3. Commit and push to `main`
4. GitHub Actions will build and deploy automatically

