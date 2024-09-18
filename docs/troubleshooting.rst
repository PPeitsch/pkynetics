Troubleshooting
===============

This guide addresses common issues you might encounter when using Pkynetics and provides solutions.

Installation Issues
-------------------

ImportError: No module named pkynetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: After installation, Python cannot find the Pkynetics module.
* **Solution**: Ensure Pkynetics is installed in the correct Python environment. Try reinstalling:

  .. code-block:: bash

     pip install --upgrade --force-reinstall pkynetics

* **Tip**: If using a virtual environment, make sure it's activated before installing and using Pkynetics.

Data Import Problems
--------------------

FileNotFoundError
^^^^^^^^^^^^^^^^^
* **Issue**: The specified data file cannot be found.
* **Solution**: Check the file path and ensure it's correct. Use absolute paths if unsure about the working directory.

ValueError: Unsupported manufacturer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: The manufacturer specified in `tga_importer` or `dsc_importer` is not recognized.
* **Solution**: Use one of the supported manufacturers: 'auto', 'TA', 'Mettler', 'Netzsch', or 'Setaram'.
* **Tip**: Try using 'auto' for automatic detection if unsure about the manufacturer.

Analysis Errors
---------------

ValueError in Kinetic Analysis Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: Errors like "invalid value encountered in log" during analysis.
* **Solution**: Check your input data for zero or negative values. Preprocess your data to handle these cases:

  .. code-block:: python

     import numpy as np
     data = np.clip(data, 1e-10, None)  # Replace zeros/negatives with a small positive value

RuntimeWarning: invalid value encountered in true_divide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: Division by zero in calculations.
* **Solution**: Ensure your temperature data is in Kelvin and conversion data is between 0 and 1. Preprocess your data:

  .. code-block:: python

     temperature_k = temperature_c + 273.15  # Convert to Kelvin if necessary
     alpha = np.clip(alpha, 0.001, 0.999)  # Avoid exact 0 or 1 values

Visualization Problems
----------------------

RuntimeError: PNG support not available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: Unable to save plots as PNG files.
* **Solution**: Ensure you have a backend that supports PNG. Install pillow:

  .. code-block:: bash

     pip install pillow

Performance Issues
------------------

Slow Performance with Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Issue**: Analysis is taking too long with large datasets.
* **Solution**: Consider downsampling your data or using more efficient data structures:

  .. code-block:: python

     # Downsample large datasets
     from scipy import signal
     downsampled_temp = signal.resample(temperature, len(temperature)//10)
     downsampled_alpha = signal.resample(alpha, len(alpha)//10)

Reporting Issues
----------------

If you encounter a bug or issue not covered here:

1. Check the `Pkynetics GitHub Issues <https://github.com/your_username/pkynetics/issues>`_ to see if it's a known problem.
2. If not, create a new issue with:
   - A minimal code example that reproduces the problem
   - The full error traceback
   - Your Pkynetics version (``print(pkynetics.__version__)``)
   - Your Python version (``python --version``)

For feature requests or general questions, consider starting a discussion in the GitHub Discussions section.
