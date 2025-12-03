"""Base class for phase-field models."""

from typing import Any
from collections.abc import Callable, Iterable
from pathlib import Path
import numpy as np
from numpy.random import default_rng
from mpi4py import MPI

from petsc4py import PETSc

import gmsh

from basix.ufl import element
from ufl import Measure
from ufl.argument import Argument
from ufl.core.expr import Expr
from ufl.integral import Integral
# from ufl.core.terminal import FormArgument
from dolfinx import fem, default_real_type
from dolfinx.io.gmsh import MeshData
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem.petsc import NonlinearProblem

from .utils import create_mesh, relativeL2error, TMat, gen_therm_noise

class ModelBase:
    """Base class for phase-field models."""
    opts: dict[str, Any]
    """A dictionary of numerical options. Initialized with default values and can be modified in subclasses.
    Contents include:

    * ``'has_thermal_noise'`` (bool): Whether to include thermal noise. Default is ``False``.
    * ``'rand_seed'`` (int): Random seed for thermal noise generation. Default is ``8347142``.
    * ``'quadr_deg'`` (int): Quadrature degree for numerical integration. Default is ``6``.
    * ``'petsc'`` (dict): A dictionary of PETSc solver options (see `this`_ for example). Default to use 
      Newton nonlinear solver with MUMPS direct linear solver.
    * ``'t_step_rtol'`` (float): Relative tolerance for time discretization error. Default is ``0.01``.
    * ``'dt_min_rescalar'`` (float): Minimum factor to reduce time step upon failure. Default is ``0.2``.
    * ``'dt_max_rescalar'`` (float): Maximum factor to increase time step upon success. Default is ``4.0``.
    * ``'dt_reducer'`` (float): Factor to reduce time step rescalar. Default is ``0.9``.
    * ``'max_successive_fail'`` (int): Maximum number of successive failures before stopping. Default is ``100``.
    * ``'min_dt'`` (float): Minimum time step size. Default is ``1e-9``.
    * ``'max_dt'`` (float): Maximum time step size. Default is ``10.0``.
    * ``'save_period'`` (int): Time step period for saving solution. Default is ``1``.
    * ``'log_file_name'`` (str): File name for logging evolution. Default is ``evolution.txt``.
    * ``'sol_file_name'`` (str): File name for saving solution. Default is ``solution.xdmf``. If the suffix is not ``.xdmf``
      or if no suffix, save solutions using :class:`dolfinx.io.VTXWriter` into a folder with the given name.
    * ``'verbose'`` (bool): Whether to print verbose output. Default is ``False``.
    
    .. _this: https://jsdokken.com/dolfinx-tutorial/chapter2/nonlinpoisson_code.html#newtons-method
    """
    params: dict[str, Any]
    """A dictionary of physical parameters. Initialized to empty dictionary and should be set in subclasses."""
    mesh_data: MeshData
    """:class:`dolfinx.io.gmsh.MeshData` object containing mesh and boundary tags. 
    Initialized to ``None`` and should be set in subclasses.
    """
    fields: dict[str, fem.Function]
    """A dictionary mapping field names to their :class:`dolfinx.fem.Function` objects. Initialized to 
    empty dictionary and should be set in subclasses.
    """
    _fields_pre: dict[str, fem.Function]
    """A dict of fields at the previous time step. Initialized to empty dict and should be set in subclasses."""
    _test_funcs: dict[str, Argument]
    _bcs: list[fem.DirichletBC]  # Dirichlet boundary conditions
    _dt: fem.Constant
    _have_dt: dict[str, bool]
    """A dict indicating whether each field has a time derivative term in its equation. 
    Initialized to empty dict and should be set in subclasses.
    """
    _noise: fem.Function
    _weak_forms: dict[str, Integral] 
    """Blocked weak form. Each entry corresponds to a field equation."""
    _problem: NonlinearProblem
    _exprs2save: dict[str, str | tuple[Expr, fem.FunctionSpace]]
    _monitor: dict[str, fem.Form]

    def __init__(self):
        """Initialize common attributes (to ``None`` if that attribute is not supposed to have meaningful value at this moment). 
        """
        self.opts = {
            "has_thermal_noise": False,
            "rand_seed": 8347142,
            "quadr_deg": 6,
            "petsc": {
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
                "snes_atol": 1e-6,
                "snes_rtol": 1e-3,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "snes_monitor": None,
            },
            "t_step_rtol": 0.01, 
            "dt_min_rescalar": 0.2, 
            "dt_max_rescalar": 4.0, 
            "dt_reducer": 0.9, 
            "max_successive_fail": 100, 
            "min_dt": 1e-9, 
            "max_dt": 10.0,
            "save_period": 1, 
            "log_file_name": "evolution.txt", 
            "sol_file_name": "solution.xdmf", 
            "verbose": False
        }
        self.params = {}
        self.mesh_data = None
        self.fields = {}
        self._fields_pre = {}
        self._test_funcs = {}
        self._bcs = []
        self._dt = None
        self._have_dt = {}
        self._noise = None
        self._weak_forms = {}
        self._problem = None
        self._exprs2save = {}
        self._monitor = {}
    
    def load_mesh(
            self, 
            comm: MPI.Comm,  
            mesh: gmsh.model | str, 
            **kwargs: Any
        ):
        """Load the mesh data.
        
        :param comm: MPI communicator.
        :type comm: MPI.Comm
        :param mesh: Gmsh model or a file name with the ``.msh`` or ``.xdmf`` format.
        :type mesh: gmsh.model | str
        :param kwargs: Additional keyword arguments passed to :py:func:`~medypt.utils.create_mesh`:

            * mesh_dim (int): Geometric dimension of the mesh. Required when ``mesh`` is a gmsh model or a ``.msh`` file.
            * rank (int): Rank of the MPI process used for generating from gmsh model or reading from ``.msh`` files.
              Required when ``mesh`` is a gmsh model or a ``.msh`` file.
            * mesh_name (str): Name (identifier) of the mesh to read from the ``.xdmf`` file. Required when ``mesh`` is
              a ``.xdmf`` file.
              
        :type kwargs: dict[str, Any]
        :returns: ``None``
        """
        self.mesh_data = create_mesh(comm, mesh, **kwargs)

    def set_bcs(self, bcs: list[tuple[str, int | Callable[[Any], Any], fem.Constant | fem.Function | np.ndarray | Callable[[Any], Any]]]):
        """Set essential and natural boundary conditions.

        :param bcs: A list of tuples specifying boundary conditions. Each tuple is ``(name, b, bc)``, where:

            * ``name`` (str): The name of the field to which the boundary condition applies.
            * ``b`` (int | Callable[[Any], Any]): The tag of or a callable defining the boundary. 
              The tag is only for boundaries with codimension 1. The callable is only for Dirichlet boundary conditions 
              and will have a calling signature ``b(x)``, where ``x`` is the spatial coordinate treated as a 2D numpy 
              array with shape (geom_dim, num_points). It should return a boolean array of shape ``(num_points,)`` 
              indicating whether each point is on the boundary.
            * ``bc`` (Constant | Function | ndarray | Callable[[Any], Any]): A variable or callable defining the boundary condition. 
              A variable is for Dirichlet boundary conditions, while a callable is for natural boundary conditions. The callable 
              will have a calling signature ``bc(fields)``, where ``fields`` is treated as a tuple containing the splitted fields. 
              It covers boundary conditions of the Neumann, Robin, and other general types implemented using the Nistche's trick 
              [1]_ [2]_. The default is the zero-flux boundary condition.

        :type bcs: list[tuple[str, int | Callable[[Any], Any], fem.Constant | fem.Function | np.ndarray | Callable[[Any], Any]]]

        .. [1] https://doi.org/10.1016/j.camwa.2022.11.025
        .. [2] https://doi.org/10.1103/PhysRevApplied.17.014042

        .. tip::

            One can make an evolving boundary condition by defining a global :py:class:`dolfinx.fem.Constant` or 
            :py:class:`dolfinx.fem.Function` object and using it in the boundary-condition expression. One will 
            then need to define an update function to update the value of the global object for a given time 
            and pass that function to :py:meth:`~medypt.model.ModelBase.solve`.
        """
        facet_dim = self.mesh_data.mesh.topology.dim - 1
        ds = Measure("ds", domain=self.mesh_data.mesh, subdomain_data=self.mesh_data.facet_tags, 
                     metadata={"quadrature_degree": self.opts["quadr_deg"]})
        self._bcs.clear()
        for name, tag, bc in bcs:
            if isinstance(bc, (fem.Constant, fem.Function, np.ndarray)):
                if callable(tag): # See https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844
                    bdr_dof = fem.locate_dofs_geometrical(self.fields[name].function_space, tag)
                else:
                    bdr_dof = fem.locate_dofs_topological(self.fields[name].function_space, facet_dim, self.mesh_data.facet_tags.find(tag))
                if isinstance(bc, fem.Function):
                    self._bcs.append(fem.dirichletbc(bc, bdr_dof))
                else:
                    self._bcs.append(fem.dirichletbc(bc, bdr_dof, self.fields[name].function_space))
            else:
                if callable(tag):
                    raise ValueError("[ModelBase.create_bcs] Natural boundary conditions cannot be defined on boundaries specified by callables.")
                if self._have_dt[name]:  # Multiply time step size for time-derivative-dependent weak forms
                    self._weak_forms[name] += self._dt * bc(self.fields) * self._test_funcs[name] * ds(tag)
                else:
                    self._weak_forms[name] += bc(self.fields) * self._test_funcs[name] * ds(tag)

    def create_problem(self):
        """Create the finite element problem that is ready to be solved."""
        names = sorted(self._weak_forms)
        F = [self._weak_forms[name] for name in names]
        u = [self.fields[name] for name in names]

        self.opts["petsc"]["ksp_error_if_not_converged"] = True # Force error if linear solver not converged
        self.opts["petsc"]["snes_error_if_not_converged"] = False # Force pass if nonlinear solver not converged; its convergence will be checked separately.
        # u parameter does not accept UFL Variable
        self._problem = NonlinearProblem(F, u, petsc_options_prefix="medypt_", bcs=self._bcs, 
                                         kind="mpi", petsc_options=self.opts["petsc"])

    def solve(
            self, 
            tspan: float, dt: float | None = None, 
            ics: dict[str, Callable[[Any], Any] | fem.Function] | None = None, 
            update: Callable[[float], None] = lambda t: None
        ) -> bool:
        """Solve the time-dependent finite-element problem.

        :param tspan: Time span to solve over.
        :type tspan: float
        :param dt: Initial time step. Defaults to ``None`` to use the current time step.
        :type dt: float | None
        :param ics: Initial conditions, defined as a dictionary mapping field names to callables that return 
            the initial value for that field. The callables have a calling signature ``ic(x)``, where ``x`` 
            is the spatial coordinate treated as a 2D numpy array with shape ``(geom_dim, num_points)``. 
            Defaults to ``None`` to use the current values of the fields.
        :type ics: dict[str, Callable[[Any], Any] | fem.Function] | None
        :param update: A callable to update any time-dependent parameters (must be defined a priori as global 
            :py:class:`dolfinx.fem.Constant` or :py:class:`dolfinx.fem.Function`) at each time step. It takes the 
            current time as the only argument. Defaults to a no-op function.
        :type update: Callable[[float], None]
        :returns: ``True`` if the solve completed successfully, ``False`` otherwise.
        :rtype: bool
        """
        if dt is not None:
            self._dt.value = dt
        if ics is not None:
            for name, ic in ics.items():
                self.fields[name].interpolate(ic)
                self.fields[name].x.scatter_forward()
                self._fields_pre[name].x.array[:] = self.fields[name].x.array

        eps = 1e-10
        p_rank = self.mesh_data.mesh.comm.Get_rank()
        verbose = (p_rank == 0) and self.opts["verbose"]
        # visualize = have_pyvista and self.opts["visualize"]
        success = True
        successive_fail = 0
        ksp_fail = 0
        snes_fail = 0
        t_fail = 0
        n_step = 0
        t = 0.0
        u2acc = {}  
        dudt_pre = {}
        for name, has_dt in self._have_dt.items():
            if has_dt:  # Only for equations with time derivative terms
                u2acc[name] = fem.Function(self.fields[name].function_space, name=f"{name}_2acc", dtype=default_real_type)
                dudt_pre[name] = fem.Function(self.fields[name].function_space, name=f"d{name}_dt_pre", dtype=default_real_type)
        log_file = open(self.opts["log_file_name"], 'w')
        folder = Path(self.opts["sol_file_name"])
        use_xdmf = (folder.suffix == ".xdmf")
        if use_xdmf:
            sol_file = XDMFFile(self.mesh_data.mesh.comm, str(folder), 'w')  # Use Xdmf3ReaderT in Paraview
            sol_file.write_mesh(self.mesh_data.mesh)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            sol_file = {}  # Store different solutions into separate files because they could be based on different finite elements
        if self.opts["has_thermal_noise"]:
            has_T = ("T" in self.fields)
            T = self.fields["T"] if has_T else (
                self.params["temperature"], 
                fem.functionspace(self.mesh_data.mesh, element("CG", self.mesh_data.mesh.basix_cell(), 1, dtype=default_real_type))
            )
            Tmat = TMat()
            Tmat.setT(T)
            if isinstance(self.params["op_relax_rate"], Iterable):
                dissipU = np.linalg.cholesky(np.asarray(self.params["op_relax_rate"]), upper=True)
            else:
                dissipU = np.sqrt(self.params["op_relax_rate"])
            rng = default_rng([p_rank, self.opts["rand_seed"]])  # Safe independent spawn of RNGs for parallel runs
            gen_therm_noise(Tmat, dissipU, rng, self._noise)
        else:
            self._noise.x.array[:] = 0.0

        # Prepare Function objects for saving the solution
        compiled_sols = {}
        funcs2save = {}  # Dictionary containing Functions for saving algebraic expressions of dolfinx Functions
        for key, expr in self._exprs2save.items():
            if isinstance(expr, str):  
                # see https://fenicsproject.discourse.group/t/typeerror-boundingboxtree-init-takes-2-positional-arguments-but-3-were-given/12825
                funcs2save[key] = self.fields[expr] # Store a convenient reference to corresponding fields
            else:
                # sols2save[key] = Projector(self.func2save[key][1], petsc_options=self.options["proj_petsc_opt"], metadata=self.options["proj_metadata"])  
                # u4save = sols2save[key].project(self.func2save[key][0])  # Returns the Function object stored in the Projector object
                # u4save.name = key
                compiled_sols[key] = fem.Expression(expr[0], expr[1].element.interpolation_points)
                funcs2save[key] = fem.Function(expr[1], name=key, dtype=default_real_type) # Create a new Function object for saving
                funcs2save[key].interpolate(compiled_sols[key]) # Interpolation does not have the overshoot issue
                # if key == self.options["sol_to_plot"]:  # This is the only case where sol2plot is needed
                #     sol2plot = u4save  # This stores a reference to the Function object in the proper Projector object
            # funcs2save[key].name = key
            if use_xdmf:
                sol_file.write_function(funcs2save[key], t)
            else:
                filename = folder / key  # Seems not distinguish between lower and upper cases
                # Set mesh_policy to reuse seems to result in empty file for collapsed Function
                sol_file[key] = VTXWriter(self.mesh_data.mesh.comm, filename.with_suffix(".bp"), funcs2save[key], engine="BP4")
                sol_file[key].write(t)
        
        # # Prepare viewer for plotting the solution during the computation
        # if visualize:
        #     vec_factor = 1.0
        #     if type(self.options["sol_to_plot"]) is int:
        #         plot_plain_sol = True
        #         plot_idx = self.options["sol_to_plot"]
        #         plot_name = u.sub(plot_idx).name
        #         # Get the sub-space and the corresponding dofs in the mixed space vector for visualizing
        #         V4plot, dofs4plot = u.function_space.sub(plot_idx).collapse()
        #         elem_dim = V4plot.dofmap.index_map_bs
        #         # Create a VTK 'mesh' with 'nodes' at the function dofs
        #         # x is the coordinates of the mesh nodes with the first dimension running over different nodes
        #         topology, cell_types, x = plot.vtk_mesh(V4plot)
        #         grid = pv.UnstructuredGrid(topology, cell_types, x)
        #         # index_map_bs (block size of the index map) is the number of degrees of freedom (DOFs) 
        #         # associated with each DOF point in a block element, i.e., the number of independent components
        #         # of the element.
        #         if elem_dim == 1:
        #             grid.point_data[plot_name] = u.x.array[dofs4plot]   # Set output data
        #         else:
        #             dat4plot = u.x.array[dofs4plot]  # Advanced slicing returns a copy
        #             grid.point_data[plot_name] = dat4plot.reshape((x.shape[0], elem_dim))   # Set output data
        #             glyphs = grid.glyph(orient=plot_name, scale=plot_name, factor=vec_factor)   # Create glyphs for the vector field

        #     elif type(self.exprs2save[self.options["sol_to_plot"]]) is int:
        #         plot_plain_sol = True
        #         plot_idx = self.exprs2save[self.options["sol_to_plot"]]
        #         plot_name = self.options["sol_to_plot"]
        #         # Get the sub-space and the corresponding dofs in the mixed space vector for visualizing
        #         V4plot, dofs4plot = u.function_space.sub(plot_idx).collapse()
        #         elem_dim = V4plot.dofmap.index_map_bs
        #         # Create a VTK 'mesh' with 'nodes' at the function dofs
        #         topology, cell_types, x = plot.vtk_mesh(V4plot)
        #         grid = pv.UnstructuredGrid(topology, cell_types, x)
        #         if elem_dim == 1:
        #             grid.point_data[plot_name] = u.x.array[dofs4plot]
        #         else:
        #             dat4plot = u.x.array[dofs4plot]
        #             grid.point_data[plot_name] = dat4plot.reshape((x.shape[0], elem_dim))
        #             glyphs = grid.glyph(orient=plot_name, scale=plot_name, factor=vec_factor)
        #     else:
        #         plot_plain_sol = False
        #         plot_name = self.options["sol_to_plot"]
        #         topology, cell_types, x = plot.vtk_mesh(self.exprs2save[plot_name][1])
        #         elem_dim = self.exprs2save[plot_name][1].dofmap.index_map_bs
        #         grid = pv.UnstructuredGrid(topology, cell_types, x)
        #         if elem_dim == 1:
        #             grid.point_data[plot_name] = funcs2save[plot_name].x.array  # sols2save[plot_name]._x.x.array
        #         else:
        #             grid.point_data[plot_name] = funcs2save[plot_name].x.array.reshape((x.shape[0], elem_dim))  # sols2save[plot_name]._x.x.array.reshape((x.shape[0], elem_dim))
        #             glyphs = grid.glyph(orient=plot_name, scale=plot_name, factor=vec_factor)
        #     grid.set_active_scalars(plot_name)
        #     plotter = pvqt.BackgroundPlotter(title=plot_name
        #                                      # , auto_update=True
        #                                      )
        #     plotter.add_mesh(grid, show_edges=True, copy_mesh=False)
        #     if elem_dim > 1:
        #         # glyphs.set_active_vectors(plot_name)
        #         # grid.set_active_vectors(plot_name)
        #         plotter.add_mesh(glyphs, show_scalar_bar=False)
        #     plotter.add_axes()
        #     plotter.view_xy(True)
        #     plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")

        if p_rank == 0:
            log_file.write("          #Step            Time       Time step   Stepping fail       SNES fail        KSP fail ")
            for key in self._monitor:
                log_file.write(f"{key:>15} ")
            log_file.write("\n")
            log_file.flush()

        while t <= tspan:
            if successive_fail > self.opts["max_successive_fail"] or self._dt.value < self.opts["min_dt"]:
                if p_rank == 0:
                    print("[ModelBase.solve] Time stepping diverged. SNES fail:", 
                          snes_fail, ", stepping fail:", t_fail, ", KSP fail:", ksp_fail, 
                          ", dt:", self._dt.value, flush=True)
                # log_file.close()
                # if use_xdmf:
                #     sol_file.close()
                # else:
                #     for key in sol_file:
                #         sol_file[key].close()
                # return False
                success = False
                break
            
            update(t + self._dt.value)    # Implicit scheme
            try:
                self._problem.solve()
            except PETSc.Error:
                successive_fail += 1
                ksp_fail += 1   # Count the number of KSP fails
                for name, func in self.fields.items():
                    func.x.array[:] = self._fields_pre[name].x.array  # Restore initial guess for Newton iteration because fields have been spoiled
                self._dt.value *= self.opts["dt_min_rescalar"]
                if verbose:
                    print(f"[ModelBase.solve] Step {n_step + 1}: KSP solver did not converge! Refined dt to {self._dt.value:15.6e} and test again", flush=True)
                continue
            
            snes_converged = self._problem.solver.getConvergedReason()
            if snes_converged <= 0: # Not converged
                successive_fail += 1
                snes_fail += 1
                for name, func in self.fields.items():
                    func.x.array[:] = self._fields_pre[name].x.array  # Restore initial guess for Newton iteration because fields have been spoiled
                self._dt.value *= self.opts["dt_min_rescalar"]
                if verbose:
                    print(f"[ModelBase.solve] Step {n_step + 1}: SNES solver did not converge! Refined dt to {self._dt.value:15.6e} and test again", flush=True)
                continue

            if n_step > 0:
                # Compute the second-order accurate estimate of the solution; the current solution u is first-order accurate (backward Euler)
                for name, u2 in u2acc.items():
                    u2.x.array[:] = (
                        self._fields_pre[name].x.array + (dudt_pre[name].x.array * self._dt.value 
                                                          + self.fields[name].x.array - self._fields_pre[name].x.array) * 0.5
                    )
                # Calculate the backward Euler time integration error and time step changing factor
                rel_err = relativeL2error(u2acc, self.fields, eps=eps)  # Only taking into account fields with time derivative terms in their equations
                dt_factor = min(max(self.opts["dt_reducer"] * np.sqrt(self.opts["t_step_rtol"] / max(rel_err, eps)), 
                                    self.opts["dt_min_rescalar"]), self.opts["dt_max_rescalar"])
                if rel_err > self.opts["t_step_rtol"]:
                    successive_fail += 1
                    t_fail += 1
                    for name, func in self.fields.items():
                        func.x.array[:] = self._fields_pre[name].x.array   # Restore initial guess for newton iteration because fields have been spoiled
                    self._dt.value *= dt_factor   # dt_factor < 1 here
                    if verbose:
                        print(f'[ModelBase.solve] Step {n_step + 1}: Time stepping error {rel_err:15.6e} exceeded tolerance, refined dt to {self._dt.value:15.6e} and test again', flush=True)
                    continue
            else:
                dt_factor = 0.5  # Set to be smaller than 1 so to skip rescaling dt for the first step

            # All tests passed, now output info and save and update solution
            n_step += 1
            t += self._dt.value

            # Save solution; To visualize in Paraview the solution file that contains multiple Functions,
            # use ExtractBlock filter for the Function to be visualized to avoid wierd behavior.
            if (n_step - 1) % self.opts["save_period"] == 0:
                for key, expr in self._exprs2save.items():
                    if isinstance(expr, tuple):
                        # u4save = sols2save[key].project(self.func2save[key][0])  # Name already set in Projector
                        funcs2save[key].interpolate(compiled_sols[key])
                    if use_xdmf:
                        sol_file.write_function(funcs2save[key], t)
                    else:
                        sol_file[key].write(t)
        
            # # Update the plot window
            # if visualize:
            #     plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
            #     if plot_plain_sol:
            #         if elem_dim == 1:
            #             grid.point_data[plot_name] = u.x.array[dofs4plot]
            #         else:
            #             dat4plot = u.x.array[dofs4plot]
            #             grid.point_data[plot_name] = dat4plot.reshape((x.shape[0], elem_dim))
            #             # glyphs = grid.glyph(orient=plot_name, factor=0.1)
            #     else:
            #         # sol2plot = sols2save[plot_name].project(self.func2save[plot_name][0])
            #         funcs2save[plot_name].interpolate(compiled_sols[plot_name])
            #         if elem_dim == 1:
            #             grid.point_data[plot_name] = funcs2save[plot_name].x.array
            #         else:
            #             grid.point_data[plot_name] = funcs2save[plot_name].x.array.reshape((x.shape[0], elem_dim))
            #             # glyphs = grid.glyph(orient=plot_name, factor=0.1)
            #     # The scalar_bar_range will not update if the data range is smaller than the scalar_bar_range 
            #     # even though we force to update by plotter.update_scalar_bar_range().
            #     # plotter.update_scalar_bar_range([0.0, 10.0])
            #     # plotter.render()
            #     # plotter.update()
            #     # print(grid.point_data[plot_name])
            #     plotter.app.processEvents()

            for name, func in dudt_pre.items():  # Store time derivative for next step
                func.x.array[:] = (self.fields[name].x.array - self._fields_pre[name].x.array) / self._dt.value  
            for name, func in self._fields_pre.items():
                func.x.array[:] = self.fields[name].x.array

            if verbose:
                print(f'[ModelBase.solve] Step {n_step}: Completed refinement, dt = {self._dt.value:15.6e}, t = {t:15.6e}', flush=True)
            if p_rank == 0:
                log_file.write(f"{n_step:15d} {t:15.6e} {self._dt.value:15.6e} {t_fail:15d} {snes_fail:15d} {ksp_fail:15d} ")
            for key, form in self._monitor.items():
                out = self.mesh_data.mesh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
                if p_rank == 0:
                    log_file.write(f"{out:15.6e} ")
            if p_rank == 0:
                log_file.write("\n")
                log_file.flush()

            successive_fail = 0
            # Reset to make these variables count #fails for current step
            ksp_fail = 0
            snes_fail = 0
            t_fail = 0

            # Increase dt for next step
            if dt_factor > 1.0:
                self._dt.value = min(dt_factor * self._dt.value, self.opts["max_dt"])
                if verbose:
                    print(f'[ModelBase.solve] Step {n_step}: dt increased to {self._dt.value:15.6e} for next step', flush=True)

            # Update thermal noise using explicit scheme, so it needs to be done after successful solving for the last time step
            if self.opts["has_thermal_noise"]:
                if has_T:
                    Tmat.assemble_lumpedT()
                gen_therm_noise(Tmat, dissipU, rng, self._noise)
        
        log_file.close()
        if use_xdmf:
            sol_file.close()
        else:
            for key in sol_file:
                sol_file[key].close()

        # # Update ghost entries and plot
        # if visualize:
        #     if plot_plain_sol:
        #         # u.x.scatter_forward()
        #         if elem_dim == 1:
        #             grid.point_data[plot_name] = u.x.array[dofs4plot]
        #         else:
        #             dat4plot = u.x.array[dofs4plot]
        #             grid.point_data[plot_name] = dat4plot.reshape((x.shape[0], elem_dim))
        #             # glyphs = grid.glyph(orient=plot_name, factor=0.1)
        #     else:
        #         # sols2save[plot_name]._x.x.scatter_forward()
        #         # funcs2save[plot_name].x.scatter_forward()
        #         if elem_dim == 1:
        #             grid.point_data[plot_name] = funcs2save[plot_name].x.array  # sols2save[plot_name]._x.x.array
        #         else:
        #             grid.point_data[plot_name] = funcs2save[plot_name].x.array.reshape((x.shape[0], elem_dim)) # sols2save[plot_name]._x.x.array.reshape((x.shape[0], elem_dim))
        #             # glyphs = grid.glyph(orient=plot_name, factor=0.1)
        #     screenshot = None
        #     if pv.OFF_SCREEN:
        #         screenshot = f"{plot_name}.png"
        #     pv.plot(grid, show_edges=True, screenshot=screenshot)

        if verbose and success:
            print("[ModelBase.solve] Time stepping done successfully", flush=True)
        return success


        