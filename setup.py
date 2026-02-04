from setuptools import setup, find_packages

setup(
    name="transpolymer-pretrained-polyagent",
    version="0.1.0",
    description="Pretrained TransPolymer models and utilities for polyagent",
    packages=find_packages(include=["transpolymer_pretrained", "transpolymer_pretrained.*"]),
    include_package_data=True,
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "pyyaml",
    ],
)

