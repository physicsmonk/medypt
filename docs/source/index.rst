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
#. Load mesh using :py:meth:`~medypt.model.ModelBase.load_mesh`
#. Set physical parameters using :py:attr:`~medypt.imt.IMTModel.params` and numerical options using :py:attr:`~medypt.model.ModelBase.opts`
#. Load physics components using :py:meth:`~medypt.imt.IMTModel.load_physics`
#. Set boundary conditions using :py:meth:`~medypt.model.ModelBase.set_bcs`
#. Create finite element problem using :py:meth:`~medypt.model.ModelBase.create_problem`
#. Solve the problem using :py:meth:`~medypt.model.ModelBase.solve`

.. _FEniCSx: https://fenicsproject.org/

Currently MEDYPT only implemented insulator-metal transition (IMT) model.
Plan to implement more phase-field models such as that of ferroelectrics in the future.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

