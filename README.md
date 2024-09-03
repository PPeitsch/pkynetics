# Pkynetics

Pkynetics is a comprehensive library for thermal analysis kinetic methods, including traditional model-fitting and model-free methods, advanced computational techniques, machine learning approaches, and result visualization.

## Early Release Version 0.1.0

This is an early release version of Pkynetics, focusing on data import capabilities and fundamental model-fitting methods. We're actively developing additional features and welcome feedback from the community.

## Features

Currently implemented:
- Data preprocessing and import from various thermal analysis instruments (TA Instruments, Mettler Toledo, Netzsch, Setaram)
- Model-fitting methods:
  - Avrami method
  - Kissinger method
  - Coats-Redfern method
  - Freeman-Carroll method
  - Horowitz-Metzger method

Planned for future releases:
- Model-free (isoconversional) methods (Friedman, OFW, KAS, Vyazovkin, etc.)
- Advanced methods (Starink, Master Plot, DAEM, etc.)
- Machine learning approaches for kinetic analysis
- Result visualization and statistical analysis
- Parallel processing capabilities

## Requirements

Pkynetics requires Python 3.8 or later. It has been tested with Python 3.8, 3.9, 3.10, and 3.11.

## Installation

You can install Pkynetics using pip:

```
pip install pkynetics
```

Make sure you have Python 3.8 or later installed on your system.

## Usage

For detailed usage instructions and examples, please refer to our documentation in the `docs/` directory and the `examples/` directory in the repository.

## Documentation

The full documentation for Pkynetics is included in the library itself, in the `docs/` directory. To build the documentation locally, navigate to the `docs/` directory and run:

```
make html
```

This will generate the HTML documentation in the `docs/_build/html/` directory. You can then open `index.html` in your web browser to view the documentation.

For online access to the documentation, please visit [our documentation site](https://pkynetics.readthedocs.io).

## Examples

You can find Jupyter notebook examples in the `examples/` directory, demonstrating various use cases and features of Pkynetics.

## Current Limitations

- Limited to specific model-fitting methods
- Visualization capabilities are not yet implemented
- Advanced kinetic analysis methods are planned for future releases
- Performance optimizations and parallel processing are under development

## Future Plans

We are actively working on expanding Pkynetics to include:
1. Implementation of model-free methods
2. Addition of advanced kinetic analysis techniques
3. Integration of machine learning approaches
4. Development of comprehensive visualization tools
5. Performance enhancements and parallel processing support
6. Expanded documentation and tutorials

We welcome contributions and feedback from the community to help shape the future of Pkynetics.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

Pkynetics is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
