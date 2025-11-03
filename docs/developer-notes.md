# Developer Notes

This page contains essential information for developers contributing to or maintaining the ada-verona package.

## Development Environment Setup

Install the package with all development dependencies using:

```bash
uv sync --dev
```

This command:
- Installs ada-verona and all dependencies
- Ensures consistency with the `uv.lock` file
- Includes all dev dependencies specified in the dependency groups in `pyproject.toml`

You can use this in your preferred virtual environment setup (conda, venv, etc.).

## Pre-Commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically enforce code quality standards and consistency before commits. 

### Installation

If you haven't already installed pre-commit, do so using:

```bash
uv pip install pre-commit
pre-commit install
```

This command hooks into your Git workflow and runs configured checks on staged files before each commit.

## Package Release Steps

### Prerequisites

Ensure you have the necessary permissions and setup for PyPI releases:

- You must be a member of the [PyPI ada-verona project](https://pypi.org/project/ada-verona/) with release authority
- Configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) with your credentials on PyPI

### 1. Bump Version Locally

Use `uv` to automatically update the version in `pyproject.toml` and `uv.lock`:

```bash
# For patch version (1.0.0 → 1.0.1)
uv version --bump patch

# For minor version (1.0.0 → 1.1.0)
uv version --bump minor

# For major version (1.0.0 → 2.0.0)
uv version --bump major
```

The package version is automatically resolved at runtime via `importlib.metadata.version("ada-verona")`.

### 2. Commit Version Changes

```bash
git add pyproject.toml uv.lock
git commit -m "Bump version to 1.0.0-alpha.10"
```

### 3. Create and Push Git Tag

**Important**: Use the `v` prefix for the git tag, as this triggers the release workflow.

```bash
# Create tag matching the version
git tag v1.0.0

# Push commit and tag together
git push origin main
git push origin v1.0.0
```

### 4. Review and Approve Workflow

The release workflow requires approval from at least one council member (see [governance.md](../governance.md)) before deployment. Monitor the [GitHub Actions](https://github.com/ADA-research/VERONA/actions) page for the workflow run and approve it as needed.




