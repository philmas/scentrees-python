"""Setup script for scentrees package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="scentrees",
    version="0.1.0",
    description="Scenario trees and lattices for multistage stochastic optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Python port from ScenTrees.jl",
    author_email="",
    url="https://github.com/kirui93/ScenTrees.jl",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "mypy>=1.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="scenario trees, scenario lattices, stochastic optimization, multistage optimization",
    project_urls={
        "Documentation": "https://kirui93.github.io/ScenTrees.jl/stable/",
        "Source": "https://github.com/kirui93/ScenTrees.jl",
        "Original Julia Package": "https://github.com/kirui93/ScenTrees.jl",
    },
)