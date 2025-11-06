"""LOBSTER Toolkit 安装配置"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lobster-toolkit",
    version="0.1.0",
    author="AEON Project",
    description="高频交易 LOBSTER 数据处理工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lobster-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.10.0"],
        "aeon": ["aeon>=0.5.0", "scikit-learn>=1.0.0"],
        "all": ["torch>=1.10.0", "aeon>=0.5.0", "scikit-learn>=1.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
