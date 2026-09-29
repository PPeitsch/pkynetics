# Pkynetics

[![PyPI version](https://badge.fury.io/py/pkynetics.svg)](https://badge.fury.io/py/pkynetics)
[![Python Versions](https://img.shields.io/pypi/pyversions/pkynetics.svg)](https://pypi.org/project/pkynetics/)
[![Documentation Status](https://readthedocs.org/projects/pkynetics/badge/?version=latest)](https://pkynetics.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/PPeitsch/pkynetics/workflows/Test%20and%20Publish/badge.svg)](https://github.com/PPeitsch/pkynetics/actions/workflows/test-and-publish.yaml)
[![Coverage](https://codecov.io/gh/PPeitsch/pkynetics/branch/main/graph/badge.svg)](https://codecov.io/gh/PPeitsch/pkynetics)
[![License](https://img.shields.io/pypi/l/pkynetics.svg)](https://github.com/PPeitsch/pkynetics/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](.github/CODE_OF_CONDUCT.md)

A Python library for thermal analysis kinetic methods, providing tools for data preprocessing, kinetic analysis, and result visualization.

## Features

### Data Import
- DSC and TGA exports from TA Instruments, Mettler Toledo, Netzsch and Setaram
- Dilatometry data
- Flexible custom importer for non-standard formats
- Automatic manufacturer detection
- Comprehensive data validation

### Analysis Methods
- Model-fitting methods:
  - Johnson-Mehl-Avrami-Kolmogorov (JMAK)
  - Kissinger
  - Coats-Redfern
  - Freeman-Carroll
  - Horowitz-Metzger
- Model-Free methods:
  - Friedman method
  - Kissinger-Akahira-Sunose (KAS)
  - Ozawa-Flynn-Wall (OFW)
- Dilatometry analysis:
  - Transformed fraction by the lever rule or the tangent method
  - Five detectors for the transformation limits (`derivative`, `offset`,
    `double_tangent`, `second_derivative`, `statistical`), chosen with `detection=`
  - Automatic baseline margin (`margin="auto"`)
  - `DilatometryAnalyzer`, which holds the analysis settings in one place
- DSC analysis: baselines, peaks, thermal events and heat capacity
- Data preprocessing, with smoothing by Savitzky-Golay, moving average or LOWESS
- Synthetic data generation for testing the kinetic methods
- Error handling and validation

### Visualization
- Comprehensive plotting functions for:
  - Kinetic analysis results
  - Dilatometry data
  - Transformation analysis
  - Custom plot styling options

## Installation

Pkynetics requires Python 3.10 or later. Install using pip:

```bash
pip install pkynetics
```

For development installation:

```bash
git clone https://github.com/PPeitsch/pkynetics.git
cd pkynetics
pip install -e .[dev]
```

For detailed installation instructions and requirements, see our [Installation Guide](https://pkynetics.readthedocs.io/en/latest/installation.html).

## Quick start

```python
from pkynetics.data import load_dilatometry_heating
from pkynetics.technique_analysis import DilatometryAnalyzer

data = load_dilatometry_heating()  # Zircaloy-4 heating run, fetched on first use

result = DilatometryAnalyzer().analyze(
    data["temperature"], data["relative_change"], method="lever"
)
print(result["start_temperature"], result["end_temperature"])
```

## Example data

Since 0.7.0 the example data does not ship with the package. `pkynetics.data` downloads
each file on first use from [pkynetics-data](https://github.com/PPeitsch/pkynetics-data),
verifies it by SHA256 and caches it on disk. Use `pkynetics.data.fetch(name)` for a path,
or one of the loaders (`load_dilatometry_heating`, `load_dsc_setaram`, `load_cp_three_step`, …)
for the data already imported. To work offline, point `PKYNETICS_DATA_DIR` at a directory
holding the files. See [Example data](https://pkynetics.readthedocs.io/en/latest/example_data.html).

## Documentation

Complete documentation is available at [pkynetics.readthedocs.io](https://pkynetics.readthedocs.io/), including:
- Detailed API reference
- Usage examples
- Method descriptions
- Best practices

## Contributing

We welcome contributions! Please read our:
- [Contributing Guidelines](.github/CONTRIBUTING.md)
- [Code of Conduct](.github/CODE_OF_CONDUCT.md)

## Security

For vulnerability reports, please review our [Security Policy](.github/SECURITY.md).

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing Pkynetics

If you use Pkynetics in your research, please cite it as:

```bibtex
@software{pkynetics,
  author = {Pablo Peitsch},
  title = {Pkynetics: A Python Library for Thermal Analysis Kinetic Methods},
  year = {2026},
  version = {0.7.0},
  publisher = {GitHub},
  url = {https://github.com/PPeitsch/pkynetics}
}
