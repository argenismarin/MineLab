Inicio Rápido
=============

Cálculo de NPV
--------------

.. code-block:: python

   from minelab.economics.cashflow import npv

   resultado = npv(rate=0.10, cashflows=[-100_000, 30_000, 35_000, 40_000, 45_000])
   print(f"NPV: ${resultado:,.0f}")

Conversión de unidades
----------------------

.. code-block:: python

   from minelab.utilities.conversions import length_convert

   metros = length_convert(100, "ft", "m")
   print(f"100 ft = {metros:.2f} m")

Clasificación de macizo rocoso
------------------------------

.. code-block:: python

   from minelab.geomechanics.rock_mass_classification import rmr_bieniawski

   rmr = rmr_bieniawski(
       ucs_rating=12,
       rqd_rating=17,
       spacing_rating=10,
       condition_rating=20,
       groundwater_rating=10,
       orientation_adj=-5,
   )
   print(f"RMR: {rmr}")
