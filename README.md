# Pkynetics

[![PyPI version](https://badge.fury.io/py/pkynetics.svg)](https://badge.fury.io/py/pkynetics)
[![Python Versions](https://img.shields.io/pypi/pyversions/pkynetics.svg)](https://pypi.org/project/pkynetics/)
[![Python Support](https://img.shields.io/pypi/pyversions/pkynetics.svg)](https://pypi.org/project/pkynetics/)
[![Documentation Status](https://readthedocs.org/projects/pkynetics/badge/?version=latest)](https://pkynetics.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/PPeitsch/pkynetics/workflows/Test%20and%20Publish/badge.svg)](https://github.com/PPeitsch/pkynetics/actions/workflows/test-and-publish.yaml)
[![Coverage](https://codecov.io/gh/PPeitsch/pkynetics/branch/main/graph/badge.svg)](https://codecov.io/gh/PPeitsch/pkynetics)
[![License](https://img.shields.io/pypi/l/pkynetics.svg)](https://github.com/PPeitsch/pkynetics/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](.github/CODE_OF_CONDUCT.md)

A comprehensive Python library for thermal analysis kinetic methods, providing robust tools for data preprocessing, kinetic analysis, and result visualization.

## Features

### Data Import
- Support for multiple thermal analysis instruments:
  - TA Instruments
  - Mettler Toledo
  - Netzsch
  - Setaram
- Flexible custom importer for non-standard formats
- Automatic manufacturer detection
- Comprehensive data validation

### Analysis Methods
- Model-fitting methods:
  - Avrami method (isothermal crystallization)
  - Kissinger method (non-isothermal kinetics)
  - Coats-Redfern method
  - Freeman-Carroll method
  - Horowitz-Metzger method
- Advanced dilatometry analysis with multiple analysis methods
- Robust data preprocessing capabilities
- Extensive error handling and validation

### Visualization
- Comprehensive plotting functions for:
  - Kinetic analysis results
  - Dilatometry data
  - Transformation analysis
  - Custom plot styling options
- Interactive visualization capabilities

## Installation

Pkynetics requires Python 3.8 or later. Install using pip:

```bash
pip install pkynetics
```

For development installation:

```bash
git clone https://github.com/PPeitsch/pkynetics.git
cd pkynetics
pip install -e .[dev]
```

## Quick Start

```python
from pkynetics.data_import import tga_importer
from pkynetics.model_fitting_methods import kissinger_method
from pkynetics.result_visualization import plot_kissinger

# Import TGA data
data = tga_importer('path/to/data.csv', manufacturer='auto')

# Perform Kissinger analysis
e_a, a, se_e_a, se_ln_a, r_squared = kissinger_method(data['temperature'], data['heating_rate'])

# Visualize results
plot_kissinger(data['temperature'], data['heating_rate'], e_a, a, r_squared)
```

## Documentation

Complete documentation is available at [pkynetics.readthedocs.io](https://pkynetics.readthedocs.io/), including:
- Detailed API reference
- Usage examples
- Method descriptions
- Best practices

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style
- Development setup
- Testing requirements
- Pull request process

## Support

- [Issue Tracker](https://github.com/PPeitsch/pkynetics/issues)
- [Discussions](https://github.com/PPeitsch/pkynetics/discussions)
- [Documentation](https://pkynetics.readthedocs.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing Pkynetics

If you use Pkynetics in your research, please cite it as:

```bibtex
@software{pkynetics2024,
  author = {Pablo Peitsch},
  title = {Pkynetics: A Python Library for Thermal Analysis Kinetic Methods},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/PPeitsch/pkynetics}
}
```

## Acknowledgments

- Contributors and maintainers
- The thermal analysis community for valuable feedback
- Open source projects that made this possible