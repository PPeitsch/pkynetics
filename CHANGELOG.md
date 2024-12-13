# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - 2024-12-13

### Added
- Code of Conduct following the Contributor Covenant
- Contributing guidelines and templates for standardizing contributions
- Security policy for vulnerability reporting
- Issue templates for bug reports, documentation issues, and feature requests 
- GitHub Actions workflow for automated testing and publishing
- New plotting functions for dilatometry data visualization
- Enhanced dilatometry analysis functions for extrapolation and transformation calculations

### Fixed
- Import path issues across examples and modules

### Changed
- Reorganized and standardized imports across all modules
- Improved code formatting consistency throughout the project
- Enhanced documentation structure and clarity
- Updated type annotations and docstrings across modules

## [v0.2.3] - 2024-11-09

### Added
- New technique_analysis module for specific thermal analysis methods
- Comprehensive dilatometry analysis capabilities
- Enhanced visualization with detailed annotations
- Automatic margin optimization for linear fitting
- Quality metrics for analysis validation

### Changed
- Reorganized code structure for better modularity
- Improved separation of preprocessing and analysis functions
- Enhanced error handling and input validation
- Better organization of helper functions
- Improved visualization capabilities

### Fixed
- Better handling of edge cases in analysis methods
- Improved accuracy in transformation point detection
- Enhanced robustness of linear segment fitting

## [v0.2.2] - 2024-10-22

### Added
- New `kissinger_nonlinear_eq` function for enhanced non-isothermal kinetics analysis
- Enhanced calculations in the Kissinger method for improved accuracy and robustness
- Five new plotting functions for better visualization of kinetic analysis data

### Changed
- Updated dependency specifications for improved functionality:
  - Removed dependencies on tensorflow and torch
  - Streamlined package requirements
- Improved data handling and filtering processes within kinetic analysis methods

### Fixed
- Improved error handling in the Kissinger method to ensure positive peak temperatures and heating rates
- Minor formatting improvements in test files for better readability

## [v0.2.1] - 2024-09-18

### Changed
- Significantly improved documentation for all modules and methods
- Enhanced clarity and structure of method descriptions
- Added more detailed usage examples and notes for each method
- Improved cross-referencing between related methods and visualization functions

## [v0.2.0] - 2024-09-17

### Added
- New plotting functions in the `result_visualization` module:
  - `plot_arrhenius`
  - `plot_conversion_vs_temperature`
  - `plot_derivative_thermogravimetry`
  - `plot_activation_energy_vs_conversion`
  - `plot_avrami_results`
- Enhanced public API for visualization functions in `result_visualization/__init__.py`

### Changed
- Improved data handling and filtering processes in kinetic analysis methods
- Streamlined plotting and data generation functions for better organization

### Fixed
- Enhanced accuracy of kinetic analysis methods through improved data handling

## [v0.1.0] - 2024-09-03

### Added
- Initial release of Pkynetics library
- Data import module:
  - Support for TGA data import from TA Instruments, Mettler Toledo, Netzsch, and Setaram
  - Support for DSC data import from TA Instruments, Mettler Toledo, Netzsch, and Setaram
  - Custom importer for flexible data import
- Model fitting methods:
  - Avrami method for isothermal crystallization kinetics
  - Kissinger method for non-isothermal kinetics analysis
  - Coats-Redfern method for kinetic analysis
  - Freeman-Carroll method for non-isothermal kinetics analysis
  - Horowitz-Metzger method for kinetic analysis
- Basic documentation and examples for each implemented method
- Unit tests for data import and model fitting methods