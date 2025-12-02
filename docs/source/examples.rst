Examples
========

Below is an example of how to set up and solve a simple phase-equilibration process in vanadium dioxide (VO2) 
using the MEDYPT library.

.. literalinclude:: ../../examples/vo2.py

Run the above script ``vo2.py`` using the command:

.. code-block:: bash

    mpirun -n 2 python3 -m mpi4py vo2.py

It will generate:

* ``evolution.txt`` file containing a summary of the phase evolution over time;
* ``solution.xdmf`` file containing the XDMF data structure of the solved spatiotemporal fields, 
  which can be visualized using tools like ParaView;
* ``solution.h5`` file containing the actual data of the fields in HDF5 format.