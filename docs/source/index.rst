.. MEDYPT documentation master file, created by
   sphinx-quickstart on Mon Nov 24 11:19:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MEDYPT documentation
====================
MEDYPT is a Python library for simulating **ME**\ soscopic **DY**\ namics of
**P**\ hase **T**\ ransitions in materials.
It uses finite element method implemented in `FEniCSx`_ to solve coupled field equations.
Typical workflow is:

#. Create mesh by users
#. Load mesh using :meth:`load_mesh`
#. Load physics components using :meth:`load_physics`
#. Set boundary conditions using :meth:`set_bcs`
#. Create finite element problem using :meth:`create_problem`
#. Solve the problem using :meth:`solve`

.. _FEniCSx: https://fenicsproject.org/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

