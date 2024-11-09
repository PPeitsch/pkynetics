from setuptools import setup, find_packages

setup(
    name="pkynetics",
    version="0.2.3",
    packages=find_packages(),
    install_requires=[
        "numpy~=1.24.3",
        "matplotlib~=3.7.5",
        "pandas~=2.0.3",
        "scipy~=1.10.1",
        "statsmodels~=0.14.1",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pylint>=2.7.0",
        ],
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
