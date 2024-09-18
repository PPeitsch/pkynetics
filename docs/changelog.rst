Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.2.1] - 2024-09-18
--------------------

Changed
^^^^^^^
- Significantly improved documentation for all modules and methods
- Enhanced clarity and structure of method descriptions
- Added more detailed usage examples and notes for each method
- Improved cross-referencing between related methods and visualization functions

[0.2.0] - 2024-09-17
--------------------

Added
^^^^^
- New plotting functions in the ``result_visualization`` module:
   - ``plot_arrhenius``
   - ``plot_conversion_vs_temperature``
   - ``plot_derivative_thermogravimetry``
   - ``plot_activation_energy_vs_conversion``
   - ``plot_avrami_results``
- Enhanced public API for visualization functions in ``result_visualization/__init__.py``

Changed
^^^^^^^
- Improved data handling and filtering processes in kinetic analysis methods
- Streamlined plotting and data generation functions for better organization

Fixed
^^^^^
- Enhanced accuracy of kinetic analysis methods through improved data handling

[0.1.0] - 2024-09-03
--------------------

Initial release of Pkynetics library

Added
^^^^^

Data import module
""""""""""""""""""
- Support for TGA data import from TA Instruments, Mettler Toledo, Netzsch, and Setaram
- Support for DSC data import from TA Instruments, Mettler Toledo, Netzsch, and Setaram
- Custom importer for flexible data import

Model fitting methods
"""""""""""""""""""""
- Avrami method for isothermal crystallization kinetics
- Kissinger method for non-isothermal kinetics analysis
- Coats-Redfern method for kinetic analysis
- Freeman-Carroll method for non-isothermal kinetics analysis
- Horowitz-Metzger method for kinetic analysis

Other
"""""
- Basic documentation and examples for each implemented method
- Unit tests for data import and model fitting methods
