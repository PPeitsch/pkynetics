# Pkynetics

Pkynetics is a Python library for thermal analysis kinetic methods. It provides tools for data preprocessing, kinetic analysis, and result visualization.

## Version 0.2.1

This version of Pkynetics includes data import capabilities, fundamental model-fitting methods, and basic visualization tools. We're actively developing additional features and welcome feedback from the community.

## Features

Currently implemented:
- Data import from various thermal analysis instruments (TA Instruments, Mettler Toledo, Netzsch, Setaram)
- Model-fitting methods:
  - Avrami method
  - Kissinger method
  - Coats-Redfern method
  - Freeman-Carroll method
  - Horowitz-Metzger method
- Basic result visualization

Planned for future releases:
- Model-free (isoconversional) methods
- Additional kinetic analysis techniques
- Machine learning approaches
- Enhanced visualization and statistical analysis
- Performance optimizations

## Requirements

Pkynetics requires Python 3.8 or later. It has been tested with Python 3.8, 3.9, 3.10, and 3.11.

## Installation

Install Pkynetics using pip:

```
pip install pkynetics
```

## Usage

For usage instructions and examples, refer to the documentation in the `docs/` directory and the examples in the `examples/` directory.

## Documentation

The full documentation is included in the `docs/` directory. To build it locally:

1. Navigate to the `docs/` directory
2. Run `make html`
3. Open `docs/_build/html/index.html` in your web browser

For online access to the documentation, visit [the Pkynetics documentation site](https://pkynetics.readthedocs.io).

## Examples

Jupyter notebook examples demonstrating various use cases are available in the `examples/` directory.

## Contributing

We welcome contributions to Pkynetics. If you're interested in contributing, please open an issue or submit a pull request on our GitHub repository.

## License

Pkynetics is released under the MIT License. See the [LICENSE](LICENSE) file for details.
