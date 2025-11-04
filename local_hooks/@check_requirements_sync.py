#!/usr/bin/env python3
"""
Pre-commit hook to ensure requirements.txt is updated when pyproject.toml changes.

This hook checks if pyproject.toml has been modified but requirements.txt has not
been updated. This helps catch cases where dependency declarations are updated
but the frozen requirements file is not regenerated.
"""

import subprocess
import sys


def get_staged_files():
    """Get list of files staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()


def main():
    """Check if pyproject.toml and requirements.txt are in sync."""
    staged_files = get_staged_files()

    pyproject_modified = "pyproject.toml" in staged_files
    requirements_modified = "requirements.txt" in staged_files

    if pyproject_modified and not requirements_modified:
        print("Error: pyproject.toml has been modified but requirements.txt has not.")
        print("Please update requirements.txt to reflect the changes in pyproject.toml.")
        print("You can regenerate it with: uv pip compile pyproject.toml -o requirements.txt")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
