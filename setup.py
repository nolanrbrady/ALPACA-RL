"""Setup script for packaging the ALPACA environment as a pip-installable module."""

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "ALPACA"
PACKAGE_DATA_PATTERNS = [
    "*.csv",
    "*.json",
    "*.joblib",
    "*.pt",
    "adni_gaussian_generation/*.csv",
]


def read_long_description() -> str:
    """Read the ALPACA README for PyPI description."""
    readme_path = BASE_DIR / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(
            f"Expected README.md for long description but none found at {readme_path}"
        )
    return readme_path.read_text(encoding="utf-8")


setup(
    name="ALPACA",
    version="0.1.0",
    description="Gymnasium environment backed by the ALPACA dynamics model for treatment optimization research.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={PACKAGE_NAME: PACKAGE_DATA_PATTERNS},
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.23",
        "pandas>=1.5",
        "torch>=2.0",
        "scikit-learn>=1.2",
        "joblib>=1.2",
    ],
)
