from setuptools import setup, find_packages

setup(
    name="pkynetics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tensorflow>=2.5.0",
        "torch>=1.8.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "pylint>=2.7.0"],
    },
    entry_points={
        "console_scripts": [
            "pkynetics=pkynetics.cli:main",
        ],
    },
    author="Pablo Peitsch",
    author_email="pablo.peitsch@gmail.com",
    description="A comprehensive library for thermal analysis kinetic methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PPeitsch/pkynetics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
