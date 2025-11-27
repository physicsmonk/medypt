from typing import Any
from collections.abc import Callable, Iterable
from pathlib import Path
import numpy as np
from numpy.random import default_rng
from mpi4py import MPI

from petsc4py import PETSc

from ufl.core.expr import Expr
from ufl.core.terminal import FormArgument
from dolfinx import fem, default_real_type
from dolfinx.io.gmsh import MeshData
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem.petsc import NonlinearProblem

from utils import relativeL2error, TMat, gen_therm_noise

class ModelBase:
    """Base class for phase-field models."""
    opts: dict[str, Any]
    """A dictionary of numerical options. Initialized with default values and can be modified in subclasses.
    Contents include:

    * ``has_thermal_noise`` (bool): Whether to include thermal noise. Default is ``False``.
    * ``rand_seed`` (int): Random seed for thermal noise generation. Default is ``8347142``.
    * ``quadr_deg`` (int): Quadrature degree for numerical integration. Default is ``6``.
    * ``petsc`` (dict): A dictionary of PETSc solver options (see `this`_ for example). Default to use 
      Newton nonlinear solver with MUMPS direct linear solver.
    * ``t_step_relative_tol`` (float): Relative tolerance for time discretization error. Default is ``0.01``.
    * ``dt_min_rescalar`` (float): Minimum factor to reduce time step upon failure. Default is ``0.2``.
    * ``dt_max_rescalar`` (float): Maximum factor to increase time step upon success. Default is ``4.0``.
    * ``dt_reducer`` (float): Factor to reduce time step rescalar. Default is ``0.9``.
    * ``max_successive_fail`` (int): Maximum number of successive failures before stopping. Default is ``100``.
    * ``min_dt`` (float): Minimum time step size. Default is ``1e-9``.
    * ``max_dt`` (float): Maximum time step size. Default is ``10.0``.
    * ``save_period`` (int): Time step period for saving solution. Default is ``1``.
    * ``log_file_name`` (str): File name for logging evolution. Default is ``evolution.txt``.
    * ``sol_file_name`` (str): File name for saving solution. Default is ``solution.xdmf``. If the suffix is not ``.xdmf``
      or if no suffix, save solutions using :class:`dolfinx.io.VTXWriter` into a folder with the given name.
    * ``verbose`` (bool): Whether to print verbose output. Default is ``False``.

    .. _this: https://jsdokken.com/dolfinx-tutorial/chapter2/nonlinpoisson_code.html#newtons-method
    """
    params: dict[str, Any]
    """A dictionary of physical parameters. Initialized to ``None`` and should be set in subclasses."""
    mesh_data: MeshData
    """:class:`dolfinx.io.gmsh.MeshData` object containing mesh and boundary tags. 
    Initialized to ``None`` and should be set in subclasses.
    """
    field: fem.Function
    """Mixed :class:`dolfinx.fem.Function` object containing all fields. Initialized to 
    ``None`` and should be set in subclasses.
    """
    field_pre: fem.Function
    """Mixed :class:`dolfinx.fem.Function` object containing all fields at the previous time step. Initialized to 
    ``None`` and should be set in subclasses.
    """
    sub_fields_ufl: dict[str, FormArgument]
    """A dictionary mapping sub-field names to their UFL representations."""
    sub_fields: dict[str, fem.Function]
    """A dictionary mapping sub-field names to their :class:`dolfinx.fem.Function` objects."""
    _sub_fields_tup: tuple[fem.Function, ...]
    _field_idx: dict[str, int]
    sub_function_spaces: dict[str, fem.FunctionSpace | tuple[fem.FunctionSpace, fem.FunctionSpace]]
    """A dictionary mapping sub-field names to their function spaces or tuples of function spaces (uncollapsed and collapsed)."""
    _sub_dof_maps: dict[str, np.ndarray]
    _bcs_natural: list[tuple[int, int, Expr]]
    _bcs_dirichlet: list[fem.DirichletBC]
    _dt: fem.Constant
    _noise: fem.Function
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
            "t_step_relative_tol": 0.01, 
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
        } # Numerical option dictionary
        self.params = None
        self.mesh_data = None
        self.field = None
        self.field_pre = None
        self.sub_fields_ufl = None
        self.sub_fields = None
        self._sub_fields_tup = None
        self._field_idx = None
        self.sub_function_spaces = None
        self._sub_dof_maps = None
        self._bcs_natural = None
        self._bcs_dirichlet = None
        self._dt = None
        self._noise = None
        self._problem = None
        self._exprs2save = None
        self._monitor = None
        
    def create_bcs(self, bcs: list[tuple[str, int | Callable[[Any], Any], fem.Constant | fem.Function | np.ndarray | Callable[[Any], Any]]]):
        """Generate boundary conditions.

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
              [10.1016/j.camwa.2022.11.025; 10.1103/PhysRevApplied.17.014042]. The default is the zero-flux boundary condition.
        :type bcs: list[tuple[str, int | Callable[[Any], Any], fem.Constant | fem.Function | np.ndarray | Callable[[Any], Any]]]

        .. tip::

            One can make an evolving boundary condition by defining a global :class:`dolfinx.fem.Constant` or 
            :class:`dolfinx.fem.Function` object and using it in the boundary-condition expression. One will 
            then need to define an update function to update the value of the global object for a given time 
            and pass that function to :meth:`ModelBase.solve`.
        """
        facet_dim = self.mesh_data.mesh.topology.dim - 1
        self._bcs_natural = []
        self._bcs_dirichlet = []
        for name, tag, bc in bcs:
            if isinstance(bc, (fem.Constant, fem.Function, np.ndarray)):
                if callable(tag): # See https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844
                    bdr_dof = fem.locate_dofs_geometrical(self.sub_function_spaces[name], tag)
                elif name == "op" or name == "u": # Treat vector function spaces properly
                    bdr_dof = fem.locate_dofs_topological(self.sub_function_spaces[name], facet_dim, self.mesh_data.facet_tags.find(tag))
                else:
                    bdr_dof = fem.locate_dofs_topological(self.sub_function_spaces[name][0], facet_dim, self.mesh_data.facet_tags.find(tag))
                self._bcs_dirichlet.append(fem.dirichletbc(bc, bdr_dof, self.sub_function_spaces[name][0]))
            else:
                if callable(tag):
                    raise ValueError("[ModelBase.create_bcs] Natural boundary conditions cannot be defined on boundaries specified by callables.")
                self._bcs_natural.append((self._field_idx[name], tag, bc(self.sub_fields_ufl)))

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
        :type ics: dict[str, Callable[[Any], Any] | Function] | None
        :param update: A callable to update any time-dependent parameters (must be defined a priori as global 
            :class:`dolfinx.fem.Constant` or :class:`dolfinx.fem.Function`) at each time step. It takes the 
            current time as the only argument. Defaults to a no-op function.
        :type update: Callable[[float], None]
        :returns: True if the solve completed successfully, False otherwise.
        :rtype: bool
        """
        if dt is not None:
            self._dt.value = dt
        if ics is not None:
            for name, ic in ics.items():
                self.sub_fields[name].interpolate(ic)
            self.field.x.scatter_forward()
            self.field_pre.x.array[:] = self.field.x.array

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
        u2acc = fem.Function(self.field.function_space)
        u2acc_subs = u2acc.split()
        dudt0 = fem.Function(self.field.function_space)
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
            Tmat = TMat()
            Tmat.setT((self.sub_fields_ufl["T"], self.sub_function_spaces["T"][1]))
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
                # Generate a fresh Function object for saving to file; 
                # see https://fenicsproject.discourse.group/t/typeerror-boundingboxtree-init-takes-2-positional-arguments-but-3-were-given/12825
                funcs2save[key] = self.sub_fields[expr].collapse()
            else:
                # sols2save[key] = Projector(self.func2save[key][1], petsc_options=self.options["proj_petsc_opt"], metadata=self.options["proj_metadata"])  
                # u4save = sols2save[key].project(self.func2save[key][0])  # Returns the Function object stored in the Projector object
                # u4save.name = key
                compiled_sols[key] = fem.Expression(expr[0], expr[1].element.interpolation_points)
                funcs2save[key] = fem.Function(expr[1]) # Create a new Function object for saving
                funcs2save[key].interpolate(compiled_sols[key]) # Interpolation does not have the overshoot issue
                # if key == self.options["sol_to_plot"]:  # This is the only case where sol2plot is needed
                #     sol2plot = u4save  # This stores a reference to the Function object in the proper Projector object
            funcs2save[key].name = key
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
                self.field.x.array[:] = self.field_pre.x.array   # Restore initial guess for Newton iteration because u has been destroyed
                self._dt.value *= self.opts["dt_min_rescalar"]
                if verbose:
                    print(f"[ModelBase.solve] Step {n_step + 1}: KSP solver did not converge! Refined dt to {self._dt.value:15.6e} and test again", flush=True)
                continue
            
            snes_converged = self._problem.solver.getConvergedReason()
            if snes_converged <= 0: # Not converged
                successive_fail += 1
                snes_fail += 1
                self.field.x.array[:] = self.field_pre.x.array   # Restore initial guess for Newton iteration because u has been destroyed
                self._dt.value *= self.opts["dt_min_rescalar"]
                if verbose:
                    print(f"[ModelBase.solve] Step {n_step + 1}: SNES solver did not converge! Refined dt to {self._dt.value:15.6e} and test again", flush=True)
                continue

            if n_step > 0:
                # Compute the second-order accurate estimate of the solution; the current solution u is first-order accurate (backward Euler)
                u2acc.x.array[:] = self.field_pre.x.array + (dudt0.x.array * self._dt.value + self.field.x.array - self.field_pre.x.array) * 0.5
                # Calculate the backward Euler time integration error and time step changing factor
                rel_err = relativeL2error(u2acc_subs, self._sub_fields_tup, eps=eps)
                dt_factor = min(max(self.opts["dt_reducer"] * np.sqrt(self.opts["t_step_relative_tol"] / max(rel_err, eps)), 
                                    self.opts["dt_min_rescalar"]), self.opts["dt_max_rescalar"])
                if rel_err > self.opts["t_step_relative_tol"]:
                    successive_fail += 1
                    t_fail += 1
                    self.field.x.array[:] = self.field_pre.x.array   # Restore initial guess for newton iteration
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
                    if isinstance(expr, str):
                        # u4save = u.sub(self.func2save[key]).collapse()   # Shallow copy seems not working properly for saving subfunctions, so use collapse()
                        # u4save.name = key
                        funcs2save[key].x.array[:] = self.field.x.array[self._sub_dof_maps[expr]]
                    else:
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

            dudt0.x.array[:] = (self.field.x.array - self.field_pre.x.array) / self._dt.value   # Store time derivative for next step
            self.field_pre.x.array[:] = self.field.x.array

            if verbose:
                print(f'[ModelBase.solve] Step {n_step}: Completed refinement, dt = {self._dt.value:15.6e}, t = {t:15.6e}', flush=True)
            if p_rank == 0:
                log_file.write(f"{n_step:15d} {t:15.6e} {self._dt.value:15.6e} {t_fail:15d} {snes_fail:15d} {ksp_fail:15d} ")
            for key in self._monitor:
                out = self.mesh_data.mesh.comm.allreduce(fem.assemble_scalar(self._monitor[key]), op=MPI.SUM)
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


        