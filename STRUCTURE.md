# Pkynetics Library Structure

```
pkynetics/
│
├── data_preprocessing/
│   ├── __init__.py
│   ├── baseline_correction.py
│   ├── smoothing.py
│   ├── peak_detection.py
│   ├── data_alignment.py
│   └── derivative_calculation.py
│
├── data_import/
│   ├── __init__.py
│   ├── tga_importer.py
│   ├── dsc_importer.py
│   └── custom_importer.py
│
├── model_fitting_methods/
│   ├── __init__.py
│   ├── avrami.py
│   ├── kissinger.py
│   ├── coats_redfern.py
│   ├── freeman_carroll.py
│   └── horowitz_metzger.py
│
├── model_free_methods/
│   ├── __init__.py
│   ├── friedman_method.py
│   ├── ozawa_flynn_wall.py
│   ├── kissinger_akahira_sunose.py
│   ├── vyazovkin_method.py
│   └── advanced_vyazovkin_method.py
│
├── advanced_methods/
│   ├── __init__.py
│   ├── starink.py
│   ├── master_plot_method.py
│   ├── distributed_activation_energy.py
│   ├── modified_coats_redfern.py
│   └── deconvolution_method.py
│
├── machine_learning_methods/
│   ├── __init__.py
│   ├── neural_network_kinetics.py
│   ├── gaussian_process_regression.py
│   ├── random_forest_kinetics.py
│   ├── svr_kinetics.py
│   └── automl_kinetics.py
│
├── result_visualization/
│   ├── __init__.py
│   ├── tg_dtg_plot.py
│   ├── kinetic_plot.py
│   ├── conversion_plot.py
│   ├── isoconversional_plot.py
│   └── ml_performance_plot.py
│
├── statistical_analysis/
│   ├── __init__.py
│   ├── confidence_intervals.py
│   ├── error_propagation.py
│   └── model_comparison.py
│
├── parallel_processing/
│   ├── __init__.py
│   ├── parallel_isoconversional.py
│   └── parallel_ml_training.py
│
├── utility_functions/
│   ├── __init__.py
│   ├── activation_energy_calculator.py
│   ├── reaction_model_selector.py
│   └── kinetic_compensation_effect.py
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_data_import.py
│   ├── test_model_fitting_methods.py
│   ├── test_model_free_methods.py
│   ├── test_advanced_methods.py
│   ├── test_machine_learning_methods.py
│   ├── test_result_visualization.py
│   ├── test_statistical_analysis.py
│   ├── test_parallel_processing.py
│   └── test_utility_functions.py
│
├── examples/
│   ├── tga_analysis_example.ipynb
│   ├── dsc_analysis_example.ipynb
│   ├── machine_learning_kinetics_example.ipynb
│   └── advanced_methods_comparison_example.ipynb
│
├── data/
│   ├── sample_tga_data.csv
│   └── sample_dsc_data.csv
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   ├── usage.rst
│   ├── api/
│   │   ├── data_preprocessing.rst
│   │   ├── data_import.rst
│   │   ├── model_fitting_methods.rst
│   │   ├── model_free_methods.rst
│   │   ├── advanced_methods.rst
│   │   ├── machine_learning_methods.rst
│   │   ├── result_visualization.rst
│   │   ├── statistical_analysis.rst
│   │   ├── parallel_processing.rst
│   │   └── utility_functions.rst
│   ├── examples/
│   │   ├── tga_analysis.rst
│   │   ├── dsc_analysis.rst
│   │   ├── machine_learning_kinetics.rst
│   │   └── advanced_methods_comparison.rst
│   └── Makefile
│
├── __init__.py
├── setup.py
├── requirements.txt
├── README.md
├── LICENSE
└── cli.py
```
