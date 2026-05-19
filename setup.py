from setuptools import setup, find_packages

setup(
    name="chembench",
    version="0.1.0",
    description="The Standardized Benchmark Suite for ML in Chemical Engineering",
    author="Mohammad Ayaan",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0"
    ],
)
