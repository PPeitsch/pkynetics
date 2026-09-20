Integral Methods: OFW and KAS
=============================

.. currentmodule:: pkynetics.model_free_methods

The integral isoconversional methods approximate the temperature integral
instead of differentiating the data, which makes them more robust to noise
than :doc:`friedman` at the cost of the approximation's own bias.

Ozawa-Flynn-Wall
----------------

Uses Doyle's approximation of the temperature integral:

.. math::

   \ln(\beta) = \ln\left(\frac{A E_a}{R g(\alpha)}\right) - 5.331 - 1.052\frac{E_a}{RT_\alpha}

At a fixed α, plotting :math:`\ln\beta` against :math:`1/T_\alpha` gives a
slope of :math:`-1.052\,E_a/R`. **The 1.052 is not optional**: dropping it
overestimates :math:`E_a` by 5 %.

Kissinger-Akahira-Sunose
------------------------

Uses the Coats-Redfern approximation, which is more accurate than Doyle's:

.. math::

   \ln\left(\frac{\beta}{T_\alpha^2}\right) = \ln\left(\frac{A R}{E_a g(\alpha)}\right) - \frac{E_a}{RT_\alpha}

Usage
-----

.. code-block:: python

   from pkynetics.model_free_methods import kas_method, ofw_method

   e_a, a, conversions, r_squared = kas_method(
       temperature_data, conversion_data, heating_rates
   )

Both take one temperature and conversion array per heating rate, and both
need **at least three heating rates**.

Pre-exponential factor
----------------------

Neither method can separate :math:`A` from :math:`g(\alpha)`, since the
intercept contains the product. What they return is the **apparent**
:math:`A/g(\alpha)` at each conversion; recovering :math:`A` itself
requires committing to a reaction model.

API
---

.. autofunction:: ofw_method

.. autofunction:: kas_method

See Also
--------

- :doc:`friedman`: The differential isoconversional method
- :doc:`../model_fitting_methods/kissinger`: The peak-temperature method
