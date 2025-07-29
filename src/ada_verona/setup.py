"""
Setup script for ada-verona package.
"""

from setuptools import setup, find_packages

setup(
    name="ada-verona",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ada-verona=ada_verona.cli:main",
        ],
    },
) 