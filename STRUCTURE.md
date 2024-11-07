# Library Structure

```
pkynetics/
│
├── data_preprocessing/
│   ├── __init__.py
│   ├── tga_preprocessing.py
│   ├── dsc_preprocessing.py
│   ├── dilatometry_preprocessing.py
│   └── common_preprocessing.py
│
├── data_import/
│   ├── __init__.py
│   ├── tga_importer.py
│   ├── dsc_importer.py
│   ├── dilatometry_importer.py
│   └── custom_importer.py
│
├── technique_analysis/
│   ├── __init__.py
│   ├── dilatometry.py
│   ├── dsc.py
│   └── tga.py
│
├── model_fitting_methods/
│   ├── __init__.py
│   ├── jmak.py
│   ├── kissinger.py
│   ├── coats_redfern.py
│   ├── freeman_carroll.py
│   └── horowitz_metzger.py
│
├── model_free_methods/
│   ├── __init__.py
│   ├── friedman_method.py
│   ├── ozawa_flynn_wall.py
│   └── kissinger_akahira_sunose.py
│
├── result_visualization/
│   ├── __init__.py
│   ├── kinetic_plot.py
│   └── model_specific_plots.py
│
├── tests/
│   ├── test_custom_importer.py
│   ├── test_data_preprocessing.py
│   ├── test_importers.py
│   ├── test_jmak_method.py
│   ├── test_kas_method.py
│   └── test_kissinger_method.py
│
├── examples/
│   ├── jmak_method_example.py
│   ├── kas_method_example.py
│   ├── dilatometry_example.py     
│   └── kissinger_method_example.py
│
├── data/
│   ├── sample_tga_data.csv
│   ├── sample_dsc_data.csv
│   ├── sample_dsc_data.txt
│   └── sample_dilatometry_data.asc    
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   ├── usage.rst
│   ├── api/
│   │   ├── data_preprocessing.rst
│   │   ├── data_import.rst
│   │   ├── technique_analysis.rst     
│   │   ├── model_fitting_methods.rst
│   │   ├── model_free_methods.rst
│   │   └── result_visualization.rst
│   ├── examples/
│   │   ├── tga_analysis.rst
│   │   ├── dsc_analysis.rst
│   │   └── dilatometry_analysis.rst   
│   └── Makefile
│
├── __init__.py
├── setup.py
├── requirements.txt
├── README.md
├── LICENSE
└── cli.py
```
