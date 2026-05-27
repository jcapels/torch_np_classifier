from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="torch_np_classifier",
    version="0.0.2",
    description="PyTorch/Lightning reimplementation of NP-Classifier.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "torch_np_classifier": [
            "data/label_names.pkl",
            "data/index_v1.json",
            "data/shap_background.csv",
        ]
    },
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit>=2022.3.1",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
)
