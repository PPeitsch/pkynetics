# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Removed
- **Breaking:** the example data no longer ships with the package. Code building a path into `pkynetics/data` will not find the files: use `pkynetics.data.fetch(name)` for a path, or one of the named loaders (`load_dilatometry_heating`, `load_dsc_setaram`, `load_cp_three_step`, …) for the data already imported. The wheel goes from 2.05 MB to under 200 KB.

### Added
- `DilatometryAnalyzer` holds the analysis settings once instead of taking them again on every call, and keeps the inputs, the ramp direction and the transformation limits reachable after `analyze()`. It returns the same mapping `analyze_dilatometry_curve` returns and calls the same code; the free functions are unchanged and are not deprecated.
- `pkynetics.data` fetches the example runs on demand from [pkynetics-data](https://github.com/PPeitsch/pkynetics-data), verifies them by SHA256 and caches them on disk, using [pooch](https://www.fatiando.org/pooch/) (a new runtime dependency). The data set has its own version, `DATA_VERSION`, independent of the library's. Point `PKYNETICS_DATA_DIR` at a directory holding the files to work offline.
- A `statistical` dilatometry detector: a departure beyond the prediction interval of the baseline regression, so its threshold has a stated meaning (a false-positive rate) rather than being a fraction someone picked. It is the most sensitive of the five and not the most accurate — on a clean curve it detects the tail of the transformation, which is a real six-sigma departure and 0.017 % of the excursion — and it does not fix the drift that limits `offset`. Since the method rests on the baseline residuals being scatter rather than structure, a Wald-Wolfowitz runs test checks that and reports a bowed baseline instead of assuming one is not there. (#26)
- A `double_tangent` dilatometry detector: the two baselines extrapolated, a tangent through the steepest part of the transformation, and the limits taken at the two crossings — the extrapolated onset of the thermal-analysis literature. It needs no threshold, and its one parameter barely matters: doubling `tangent_window_fraction` moves the limits by about a kelvin. On the Zry-4 heating run it returns 860-926 degC, recovering the 855-935 that used to be the reference value as a measurement rather than a reading off a plot. (#26)
- Two more dilatometry detectors, selected with `detection`. `offset` takes a fixed departure from the extrapolated baseline, the analogue of the 0.2 % offset of a tensile test; it is reproducible between runs but systematically conservative, and it drifts on a bowed baseline — the Zry-4 heating run fits its baseline at R² = 0.999 and still departs from it by 1.3 % of the transformation excursion, enough for a 2 % offset to report the start at 722 degC against a real ~839, so the detector measures that floor and warns when the offset does not clear it. `second_derivative` takes the extrema of `d2S/dT2`, which are the points of maximum curvature and therefore sit inside the feet of the transformation rather than at them. (#26)
- Dilatometry takes a `detection` parameter, separate from `method`, that picks how the transformation limits are located. The two answer different questions — `detection` is *where* the transformation is, `method` is *how far it has gone* — and cross freely. `detection="derivative"`, the default, is what the module already did, so nothing changes for callers who leave it alone. `available_detectors()` lists the names it accepts, and `get_detector()` hands one back to use on its own. (#26)
- Dilatometry results report `max_backward_step`: the largest step the transformed fraction takes backwards, as a fraction of full scale. The fraction is a point-by-point reading of the strain and is not constrained to be monotonic, so noise can push it back — 1.03 % on the shipped heating run, 0.28 % on the cooling one. It is reported rather than smoothed away, and the tangent method records a warning in `fit_quality["warnings"]` above 5 %. (#103)
- `find_transformation_limits` returns a `TransformationLimits` named tuple with `start_idx` and `end_idx`, and `calculate_fit_quality` returns a `FitQuality` dataclass. Both are new names for values that were already there: the limits still unpack as a pair, and the analysis result still carries the fit quality as a mapping with the same keys. (#103)
- Tests that need an example run carry a `network` marker; `pytest -m "not network"` runs the rest, and a CI job does exactly that so the suite stays usable without a network.

### Fixed
- Dilatometry located the transformation limits on a derivative differenced point to point, whose noise grows as the sample spacing shrinks. On a cooling run recorded every 0.25 K that noise reached 7% of the peak excursion of the derivative and set the detection threshold, so the limits came back 30 and 110 K inside the transformation: 915-870 degC against a contraction running from ~945 to ~760. `dS/dT` now comes from a local polynomial fit, which uses every point in the window instead of two, and the same run returns 938-757 degC. The fit differentiates against the sample index, so it is divided by `dT/di`: a ramp that changes rate no longer looks like a change in the strain. Points where the ramp stalls, whose `dS/dT` is undefined, are interpolated over rather than becoming the peak the thresholds are measured against.
- Dilatometry reported transformation limits that were not the transformation. Both methods located them by the deviation of the strain from an extrapolated tangent, with a threshold set to three standard deviations of the residuals *inside the fitting window* — a measure of how straight the baseline is, not of how large the transformation is. On a clean curve that threshold collapses to ~1e-9 and the first point examined already clears it, so `tangent` returned the first and last temperature of the data and `lever` returned the edges of its search window. The limits now come from the derivative `dS/dT`, where the curvature of a real baseline stays small and the transformation is a large localised excursion. On the Zry-4 run shipped with the package both methods now put the alpha->beta contraction at 840-929 degC, against 703-1000 and 741-889 before; on a synthetic 705-795 degC transformation both return 707-793. (#94)
- `find_optimal_margin` returned the margin with the highest R², which is always the narrowest one, since a shorter window fits a line more easily. It now returns the widest margin whose fits both reach `min_r2`, which is what makes the baselines representative.
- The test suite used `np.trapezoid`, which needs numpy 2.0, while `pyproject.toml` declares `numpy>=1.24.3`: four DSC tests failed on any numpy 1.x, a version the project claims to support. They now use `scipy.integrate.trapezoid`, as the rest of the package does.

### Changed
- `technique_analysis/dilatometry.py` is now a package: `core`, `transformation_points`, `linear_segments`, `transformed_fraction`, `methods/{lever,tangent}`, `curve_features`, `utilities` and `types`, following the layout of `technique_analysis/dsc`. Nothing moved out of reach — every public name is importable from `pkynetics.technique_analysis.dilatometry` and from `pkynetics.technique_analysis` exactly as before, and the test suite that covers them was not touched.
- **Breaking:** `analyze_dilatometry_curve` and `tangent_method` take `deviation_fraction` (default 0.05, the fraction of the peak derivative excursion that still counts as transforming) in place of `deviation_threshold`, which no longer has a meaning. `fit_quality` and `parameters` report `deviation_fraction` instead of `deviation_threshold` for the same reason.
- **Breaking:** `find_inflection_points` takes `deviation_fraction` instead of `residual_std_multiplier` and `min_points_fit`, and its `margin` default drops from 0.3 to 0.2. `find_transformation_points` and `calculate_deviation_threshold` are gone, replaced by `find_transformation_limits`.

### Removed
- The `skills/` submodule, a private repository holding the maintainers' GitHub and release workflows. Nothing in the library, the tests, CI or the packaging referenced it, and it only made `git clone --recursive` fail for anyone without access.

### Added
- `find_transformation_limits` locates a transformation on the derivative of the strain and is shared by both dilatometry methods, so `lever` and `tangent` no longer disagree about where the transformation is.
- A `minimum-versions` job in CI installs the declared dependency floors (numpy 1.24.3, pandas 2.0.3, scipy 1.10.1, matplotlib 3.7.5, statsmodels 0.14.1, chardet 5.0.0) and runs the suite against them. `pip install .` always resolves to the newest release, so nothing exercised the lower bounds and they could drift from what the code actually needs.


## [v0.6.0] - 2026-09-20

### Added
- Dilatometry, its utilities and all fifteen plotting helpers now have tests; overall coverage goes from 68% to 86%.
- Sphinx documents `technique_analysis` (DSC and dilatometry), the Friedman and OFW/KAS model-free methods, and `result_visualization`, none of which had pages before.
- A Documentation job in CI builds the docs with `sphinx-build -W`, so a missing page or a malformed docstring fails there.
- `BaselineCorrector.correct` accepts `step_regions` and fits the baseline on each side of a step, interpolating within it, the way ASTM and ISO draw it.
- Setaram exports that do not name the manufacturer in their header are detected from their columns, so `dsc_importer(..., manufacturer="auto")` handles the bundled Setaram files.

### Changed
- **Breaking:** `DSCAnalyzer.analyze()` locates a glass transition on the raw curve and fits a stepped baseline around it by default, so its results change for any run containing one. Pass `detect_steps=False` for the previous single-baseline behaviour.
- **Breaking:** `horowitz_metzger_method`, `horowitz_metzger_plot` and `plot_horowitz_metzger` take `heating_rate` (K/min) as their third positional argument. The pre-exponential factor cannot be computed without it.
- The spline baseline's `smoothing` is now relative to the noise rather than an absolute residual budget, so the same value means the same thing on any signal and at any sampling rate. Baselines fitted with `method="spline"` change, and results reported in different heat-flow units now agree.
- `SignalProcessor.remove_outliers` scores points with the median and MAD instead of the mean and standard deviation, so it flags outliers the previous score could not reach.
- `PeakAnalyzer.find_peaks` raises its prominence and height floors to at least five times the noise left after smoothing, so noisy data no longer yields spurious peaks.
- The published changelog is generated from `CHANGELOG.md` instead of a hand-maintained copy, and `docs/requirements.txt` follows `pyproject.toml` instead of pinning numpy 1.24 / pandas 2.0.
- The remaining hard-coded gas constants use `scipy.constants.R`.

### Fixed
- Horowitz-Metzger computed the pre-exponential factor from the intercept of its plot, which carries no information about it, and without the heating rate: the same curve analysed at 5 or 20 K/min returned the same A, over two orders of magnitude off. It now follows from the rate maximum at T_s. Note that the method's own approximation still biases E_a about 11% high, and A is exponential in that error; this is documented.
- `remove_outliers` could not detect outliers in the truncated windows at either end of a signal, or three sharing one window: the mean/std z-score is bounded by (n-1)/sqrt(n) and never reached the threshold.
- The spline baseline gave different results for the same experiment reported in mW and in uW, and tightened as points were added.
- With noise at 5% of the peak amplitude, about 1% of runs reported a second, spurious peak.
- A glass transition no longer distorts the events after it: a baseline fitted across the whole curve cut through the step, and a polymer melt read 0.82 J/g against a true ~19 J/g.
- The bundled example scripts resolve the sample data through the installed package instead of a relative path, so they run from any working directory.
- `installation.rst` documented a `[full]` extra that does not exist.


## [v0.5.0] - 2026-09-19

### Added
- DSC visualization (`plot_dsc_curve`, `plot_thermal_events`, `plot_cp`, `plot_dsc_analysis`).
- `reference_cp()`: NIST-JANAF Shomate reference data for sapphire and zinc, with valid ranges.
- Importers read TA Instruments Universal Analysis text exports (detected automatically).
- Examples: eicosane melting (TA data) and stepped Cp on the bundled Setaram runs.
- CI builds the sdist and wheel on every push and PR and runs the tests against the installed wheel; releases publish exactly those files.
- Tests for Coats-Redfern and the synthetic data generator; KAS, OFW and Kissinger tests check the recovered activation energy.

### Changed
- **Breaking:** Python 3.10 or later is required (3.9 is end of life). Tested on 3.10-3.13.
- **Breaking:** `kissinger_method` takes peak temperatures in kelvin, like every other function (it took °C). Passing °C does not raise an error: add 273.15 first.
- **Breaking:** `generate_basic_kinetic_data` integrates the rate equation over temperature, so the generated curves change (see Fixed).
- KAS and OFW document their pre-exponential output as the apparent A/g(α).
- Upper bounds on dependencies (`numpy<3`, `pandas<4`, `scipy<2`, `matplotlib<4`, `statsmodels<0.16`, `chardet<8`); `scikit-learn` and `seaborn` are no longer dependencies (unused).
- **Breaking:** DSC enthalpies are in J/g and require `heating_rate` (K/min) and `sample_mass` (mg); they are NaN otherwise. Peak onset/endset follow the ISO 11357-1 extrapolated definition.
- **Breaking:** `CpCalculator` STEPPED and MODULATED modes require the `time` array; the blank run is passed as `blank_heat_flow`; `exo_up` sets the sign convention.
- **Breaking:** `StabilityMethod` has two members, `STATISTICAL` and `LINEAR_FIT`.
- **Breaking:** `DataValidator.check_sampling_rate` takes only the time array; `detect_temperature_program` returns `end_idx` and NaN average rates for absent segment types.
- `ThermalEventDetector.detect_events` always returns every key (empty lists) and accepts `heating_rate`, `sample_mass` and `exo_up`.
- `DSCAnalyzer` finds peaks in both directions and reports enthalpies in J/g.
- `pkynetics.technique_analysis` exports the `dsc` subpackage.

### Fixed
- The sdist contained no modules (every release since 0.4.5): installing from source gave an empty package.
- KAS activation energy was multiplied by R (1245 kJ/mol for 150); OFW multiplied by Doyle's 1.052 instead of dividing (+11 %); Coats-Redfern returned kJ/mol labelled J/mol and a wrong pre-exponential factor.
- `from pkynetics.model_free_methods import ofw_method` returned the module instead of the function.
- `generate_basic_kinetic_data` used `1 - exp(-k(T)·t)` with A in 1/s and t in minutes instead of non-isothermal kinetics.
- `dilatometry_importer` failed on decimal-comma files with pandas 3 and dropped the first data row.
- Single-step Cp was 60x too small (heating rate in K/min); three-step Cp did not subtract the blank; the sapphire (+43%) and zinc (+21%) reference data were wrong.
- Stepped Cp integrates the heat of each heating step (step method) instead of averaging heat flow.
- Signal stability detectors: the enum exported by the package raised "Unknown stability detection method", and the detectors returned regions shorter than `min_points`.
- Baselines: ALS used a dense N x N matrix (~14 GB for 19k points); `auto` always picked `linear` without regions; rubberband and quiet-region selection.
- Temperature program detection compared K/s against a K/min threshold; `remove_outliers` produced NaN on flat regions.
- Encoding detection: chardet 7 misdetects BOM-less UTF-16 and Latin-1 files, so the bundled sample files could not be imported.
- Setaram header row located by content: the heat capacity runs imported with every column set to None.


## [v0.4.8] - 2026-03-05

### Changed
- Replaced local .agents/skills/ directory with the remote PPeitsch/skills Git submodule.
- Renamed AGENTS.md to AGENT.md to align with AI editor conventions.
- Enforced strict code quality pipeline checks prior to commits in AGENT.md.


## [v0.4.7] - 2026-02-23

### Added
- Automated GitHub Releases generation extracting notes from the project's CHANGELOG.md upon release tag pushes.


## [v0.4.6] - 2026-02-23

### Added
- New skills for automated AI interaction with GitHub CLI (Issue and PR parsing, creation and updating).
- New skills for automated and safe releases (safe tag push and changelog dynamic building).
- Redefined project contribution conventions, replacing bracket type formatting (`[ADD]`) with the Conventional Commits standard (`feat:`).


## [v0.4.5] - 2026-02-21

### Added
- Comprehensive examples for the DSC analysis module (`stepped_isothermal_cp_example.py`, `dsc_heat_capacity_example.py`, `dsc_baseline_comparison_example.py`, `polymer_analysis_example.py`, `dsc_smoothing_effects_example.py`) demonstrating capabilities on both synthetic and real analytical data.

### Fixed
- Fixed an architectural flaw in `CpCalculator` where pure isothermal holds were incorrectly validated as "stable ramps" resulting in artificially zeroed Cp measurements for the stepped-isothermal DSC method. Ramps are proactively filtered to guarantee a minimum variance.
- Addressed `TypeError` in Kissinger's root-finding evaluation caused by Numpy 2 dropping 1D array scalar casting support.
- Fixed strictly-typed `mypy` assignment assertions across `signal_stability.py` and Matplotlib plotting utilities to fully comport with static CI sub-typing restrictions under rigorous Python 3.11/Numpy 2 environments.
- Corrected `CustomImporter` instantiation parameters to properly import `utf-16le` 6-column tabular instrument data without registering `NaN`s in DSC examples.
- Improved Stepped Cp graphing logic within examples to accurately overlap theoretical vs empirical representations regardless of dataset density differences.


## [v0.4.4] - 2025-12-11

### Changed
- Updated `scipy` and `numpy` dependencies to use `>=` instead of `~=` for more flexible version compatibility.
- Replaced deprecated `np.trapz` with `np.trapezoid` for scipy 1.14+ compatibility.
- Replaced deprecated pandas `delim_whitespace` parameter with `sep=r"\s+"`.

### Fixed
- Resolved all mypy type checking errors for stricter numpy/matplotlib type stubs.
- Added explicit type casts using `typing.cast()` for numpy array operations.
- Fixed matplotlib Figure type handling in `model_specific_plots.py`.
- Added `.gitattributes` file for consistent line endings across platforms.

### Added
- `AGENTS.md` and `WORKFLOW.md` documentation files.


## [v0.4.3] - 2025-07-29

### Security
- Patched `setuptools` vulnerability GHSA-r9hx-vwmv-q579 by upgrading the dependency to version `>=78.1.1` in development and documentation requirement files. Resolves #75.


## [v0.4.2] - 2025-07-17

### Changed

-   Reorganized the `examples/` directory into categorized subdirectories (`dsc/`, `dilatometry/`, `kinetic_methods/`, etc.) to improve structure and user navigation.
-   Renamed the generic `dsc_example.py` to `legacy_dsc_analysis_example.py` and simplified its content to serve as a basic demonstration.


## [v0.4.1] - 2025-07-15

### Fixed

-   Resolved intermittent failures in `test_find_peaks_with_noise` by enhancing the robustness of the `PeakAnalyzer`. The peak detection algorithm now uses adaptive thresholds for prominence and height, calculated based on the signal's noise level, which prevents false positives from random fluctuations.


## [v0.4.0] 2025-07-15

### Added

-   **Comprehensive DSC Analysis Module (`technique_analysis.dsc`)**:
    -   `DSCAnalyzer`: A core class to orchestrate the complete analysis workflow for DSC data.
    -   `BaselineCorrector`: Implemented multiple baseline correction methods, including linear, polynomial, spline, asymmetric least squares, and rubberband, along with an automatic optimization feature.
    -   `PeakAnalyzer`: Developed for robust peak detection, characterization (onset, endset, height, width), and deconvolution of overlapping peaks.
    -   `ThermalEventDetector`: Added functionality to identify and characterize key thermal events such as glass transitions (Tg), crystallization, and melting, with refined logic to differentiate between event types in complex signals.
    -   `CpCalculator`: Implemented a full-featured calculator for specific heat capacity (Cp) with support for three-step, single-step, and modulated methods, including uncertainty propagation and reference material calibration.
    -   `SignalStabilityDetector`: Created a tool to identify stable signal regions using various methods (derivative, statistical, linear fit, etc.), essential for stepped Cp calculations.
    -   A full suite of data types (`types.py`) and utility functions (`utilities.py`) to support the DSC analysis workflow.
-   **New Example Script**: Added `dsc_example.py` to demonstrate a complete analysis workflow using the new module.
-   **Comprehensive Test Suite**: Added extensive tests for all new DSC analysis components, ensuring robustness, accuracy, and correct error handling.

### Fixed

-   Resolved numerous `mypy` static type checking errors across the new DSC module, improving code quality and maintainability.
-   Fixed performance issues and infinite loops in `SignalStabilityDetector` and `BaselineCorrector` that caused tests to hang.
-   Refined peak and event detection algorithms to prevent false positives on noisy data and correctly handle complex, overlapping thermal events.


## [v0.3.6] - 2025-03-29

### Added
- New `detect_segment_direction` function to determine if data is from cooling or heating segments
- Enhanced dilatometry analysis to properly handle cooling segments (decreasing temperature)
- Direction indicators in visualizations to clearly show cooling vs heating segments
- Improved command-line options for batch processing in dilatometry analysis

### Changed
- Modified `find_inflection_points` to adjust behavior based on segment direction
- Updated `calculate_transformed_fraction_lever` to handle both cooling and heating
- Enhanced lever rule and tangent methods to work with both heating and cooling directions
- Improved error messages and validation for temperature ranges

### Fixed
- Fixed temperature validation that previously failed for cooling segments
- Enhanced error handling for insufficient data points
- Improved temperature range validation to accommodate both heating and cooling segments


## [v0.3.5] - 2025-03-11

### Added
- Added `reaction_model` parameter to `generate_coats_redfern_data` function

### Fixed
- Fixed division by zero error in nth_order reaction model when n=1
- Fixed NaN values in pre-exponential factor calculation in Coats-Redfern method


## [v0.3.4] - 2025-03-10

### Added
- New `horowitz_metzger_plot` function in model_fitting_methods module
- New `plot_horowitz_metzger` function in result_visualization module

### Fixed
- Fixed missing imports in examples/horowitz_metzger_method_example.py
- Fixed references in documentation to non-existent functions


## [v0.3.3] - 2024-12-27

### Added
- Enhanced DSC data import functionality for Setaram file formats
- Support for multiple delimiter and decimal separator styles
- Comprehensive file format detection for Setaram DSC files

### Changed
- Improved robustness of data import mechanisms
- Enhanced logging for file format detection
- Maintained backwards compatibility with existing import methods

### Fixed
- Better handling of file import variations
- Improved error handling for different file configurations


## [v0.3.2] - 2024-12-27

### Changed
- Improved GitHub issue template user experience
- Relocated guidelines checkbox section in issue templates
- Enhanced template layout for more intuitive issue creation workflow

### Fixed
- Restructured issue template to reduce friction in reporting process
- Maintained all existing guideline compliance checks


## [v0.3.1] - 2024-12-26

### Fixed
- GitHub Actions permissions causing Codecov testing and badge updates to fail

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
