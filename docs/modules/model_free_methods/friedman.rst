Friedman Method
===============

.. currentmodule:: pkynetics.model_free_methods

The Friedman method is the differential isoconversional method: at each
conversion α it fits the reaction rate across several heating rates,
without assuming a reaction model.

Theory
------

.. math::

   \ln\left(\beta \frac{d\alpha}{dT}\right) = \ln[A f(\alpha)] - \frac{E_a}{RT_\alpha}

At a fixed α, plotting :math:`\ln(\beta\, d\alpha/dT)` against
:math:`1/(RT_\alpha)` over the heating rates gives a straight line whose
slope is :math:`-E_a`. Repeating it across α gives the activation energy
as a function of conversion, which is what reveals a multi-step process:
an :math:`E_a` that varies with α is the signature of one.

Being differential, it makes no approximation of the temperature integral
— unlike KAS or OFW — but it is correspondingly sensitive to noise in
:math:`d\alpha/dT`.

Usage
-----

.. code-block:: python

   from pkynetics.model_free_methods import friedman_method

   # One temperature and conversion array per heating rate
   activation_energies, pre_exp_factors, conversions, r_squared = friedman_method(
       temperature_data, conversion_data, heating_rates
   )

Requires **at least three heating rates** for a meaningful fit.

API
---

.. autofunction:: friedman_method

See Also
--------

- :doc:`ozawa_flynn_wall`: The integral isoconversional methods (OFW, KAS)
- :doc:`../model_fitting_methods/index`: Methods that do assume a reaction model
