Instalación
===========

Requisitos
----------

- Python >= 3.10
- NumPy, SciPy, Pandas, Matplotlib, Plotly

Instalación con pip
-------------------

.. code-block:: bash

   pip install minelab

Instalación para desarrollo
----------------------------

.. code-block:: bash

   git clone https://github.com/minelab/minelab.git
   cd minelab
   py -m uv sync --extra dev

Ejecutar tests
--------------

.. code-block:: bash

   py -m uv run pytest tests/ -v --tb=short

Lint y formato
--------------

.. code-block:: bash

   py -m uv run ruff check src/
   py -m uv run ruff format src/
