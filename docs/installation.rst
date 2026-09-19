Installation
============

Pkynetics requires Python 3.10 or later. It is tested with Python 3.10, 3.11, 3.12 and 3.13.

Basic Installation
------------------

You can install Pkynetics using pip:

.. code-block:: bash

    pip install pkynetics

Make sure you have Python 3.10 or later installed on your system.

Dependencies
------------

Pkynetics has the following core dependencies:

- numpy>=1.24.3,<3
- pandas>=2.0.3,<4
- scipy>=1.10.1,<2
- matplotlib>=3.7.5,<4
- statsmodels>=0.14.1,<0.16
- chardet>=5.0.0,<8

These will be automatically installed when you install Pkynetics using pip.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

For advanced features and improved performance, you may want to install the following optional dependencies:

- scikit-learn>=1.0.2 (for machine learning-based methods)
- numba>=0.56.4 (for performance optimizations)

To install Pkynetics with all optional dependencies, use:

.. code-block:: bash

    pip install pkynetics[full]

Development Installation
------------------------

For developers who want to contribute to Pkynetics, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/your_username/pkynetics.git
    cd pkynetics
    pip install -e .[dev]

This will install all development dependencies, including pytest for running tests.

Verifying Installation
----------------------

After installation, you can verify that Pkynetics is correctly installed by running:

.. code-block:: python

    import pkynetics
    print(pkynetics.__version__)

This should print the version number of Pkynetics without any errors.

Troubleshooting
---------------

If you encounter any issues during installation, please refer to the :doc:`troubleshooting` guide.
