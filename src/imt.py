from collections.abc import Callable, Iterable
from typing import Any
import numpy as np

import gmsh

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, default_real_type
# from dolfinx.common import Timer, timing # list_timings, TimingType
from dolfinx.fem.petsc import NonlinearProblem

from scifem import create_real_functionspace

from mpi4py import MPI

import model
import utils


class IMTModel(model.ModelBase):
    """Class for simulating mesoscopic dynamics of insulator-metal transitions in materials.

    The sub-fields are:

    * ``op``: Order parameter (vector field)
    * ``u``: Displacement (vector field)
    * ``ge``: Reduced chemical potential for electrons (scalar field). Equivalent to specifying electron density.
      For definition, see the note in :meth:`IMTModel.create_problem`.
    * ``gh``: Reduced chemical potential for holes (scalar field). Equivalent to specifying hole density. For definition,
      see the note in :meth:`IMTModel.create_problem`.
    * ``phi``: Electrical potential (scalar field)
    * ``T``: Temperature (scalar field)
    * ``j0``: Helper constant field representing the *average* outward current density on the boundary tagged by 0. Boundary
      current is ``j0`` multiplied by the area of the boundary. If you need to use boundary current in defining 
      your problem, make sure to tag the corresponding boundary with 0 in the mesh. This field only involves 
      negligible overhead if not used.

    Units used in this class are:

    * Length: nm
    * Time: ns
    * Energy: eV (temperature absorbs the Boltzmann constant and is also in this unit)
    * Charge: e (positive elementary charge)
    """
    def __init__(self):
        """Initialize the parameters, options, and field names.
        """
        super().__init__()
        self.params = {
            "stiffness": [140.0 / 1.60217663e-1, 0.3], # 1D array-like: Young's modulus in unit of eV / nm^3 and Poisson's ratio (isotropic); 2D array-like: full tensor in Voigt notation
            "e_eff_mass": 1.0, 
            "h_eff_mass": 1.0,
            "therm_expan_coeff": 0.1, # 1 / eV. Scalar: isotropic; 1D array-like: full vector in Voigt notation
            "therm_expan_Tref": 0.025, # eV
            "op_relax_rate": 100.0, # nm^3 / (eV ns). Scalar: isotropic; 2D array-like: full tensor
            "e_mobility": 5e4, # nm^2 / (V ns). Scalar: isotropic; 2D array-like: full tensor
            "h_mobility": 5e4,  # nm^2 / (V ns). Scalar: isotropic; 2D array-like: full tensor
            "eh_recomb_rate": 0.1, # nm^3 / ns
            "back_diel_const": 10.0, # Scalar: isotropic; 2D array-like: full tensor
            "vol_heat_cap": 217.0, # nm^-3
            "therm_conduct": 4.3e5 # nm^-1 ns^-1. Scalar: isotropic; 2D array-like: full tensor
        } # Physical parameter dictionary
        """A dictonary defining physical parameters. Contents include:

        * ``stiffness`` (array-like, eV / nm^3): Stiffness tensor. 1D array-like (containing Young's modulus and Poisson's ratio: isotropic) or 
          2D array-like (full tensor in Voigt notation)
        * ``e_eff_mass`` (float, dimensionless): Electron effective mass devided by free electron mass
        * ``h_eff_mass`` (float, dimensionless): Hole effective mass devided by free electron mass
        * ``therm_expan_coeff`` (float | array-like, eV^-1): Thermal expansion coefficient. Scalar (isotropic) or 
          1D array-like (full vector in Voigt notation)
        * ``therm_expan_Tref`` (float, eV): Reference temperature for thermal expansion
        * ``op_relax_rate`` (float | array-like, nm^3 / (eV ns)): Order-parameter relaxation rate. Scalar (isotropic) or 2D array-like (full tensor)
        * ``e_mobility`` (float | array-like, nm^2 / (V ns)): Electron mobility. Scalar (isotropic) or 2D array-like (full tensor)
        * ``h_mobility`` (float | array-like, nm^2 / (V ns)): Hole mobility. Scalar (isotropic) or 2D array-like (full tensor)
        * ``eh_recomb_rate`` (float, nm^3 / ns): Electron-hole recombination rate
        * ``back_diel_const`` (float | array-like, dimensionless): Background dielectric constant. Scalar (isotropic) or 2D array-like (full tensor)
        * ``vol_heat_cap`` (float, nm^-3): Volumetric heat capacity
        * ``therm_conduct`` (float | array-like, nm^-1 ns^-1): Thermal conductivity. Scalar (isotropic) or 2D array-like (full tensor)
        """
        self._field_idx = {"op": 0, "u": 1, "ge": 2, "gh": 3, "phi": 4, "T": 5} # Indices must start from 0 and be continuous

    def create_fields(
            self, 
            comm: MPI.Comm, 
            op_dim: int, 
            mesh: gmsh.model | str, 
            mesh_dim: int = 3, 
            rank: int = 0, 
            mesh_name: str = "mesh"
        ):
        """Generate the function space and fields.

        :param comm: MPI communicator.
        :type comm: MPI.Comm
        :param op_dim: Dimension of the 1D order-parameter array.
        :type op_dim: int
        :param mesh: Gmsh model or a file name with the ``.msh`` or ``.xdmf`` format.
        :type mesh: gmsh.model | str
        :param mesh_dim: Geometric dimension of the mesh.
        :type mesh_dim: int
        :param rank: Rank of the MPI process (used for generating from gmsh model or reading from ``.msh`` files).
        :type rank: int
        :param mesh_name: Name (identifier) of the mesh to read from the ``.xdmf`` file.
        :type mesh_name: str
        """
        self.mesh_data = utils.create_mesh(comm, mesh, mesh_dim, rank=rank, mesh_name=mesh_name)

        FE_P1 = element("CG", self.mesh_data.mesh.basix_cell(), 1, dtype=default_real_type)
        FE_P1_VEC = element("CG", self.mesh_data.mesh.basix_cell(), 1, shape=(mesh_dim,), dtype=default_real_type)
        FE_P1_VEC2 = element("CG", self.mesh_data.mesh.basix_cell(), 1, shape=(op_dim,), dtype=default_real_type)
        # FE_R0 = element("R", self.mesh_data.mesh.basix_cell(), 0, dtype=default_real_type)
        ME = mixed_element([FE_P1_VEC2, FE_P1_VEC, FE_P1, FE_P1, FE_P1, FE_P1])

        FS = fem.functionspace(self.mesh_data.mesh, ME)
        # Create real function space, which contains only one real number uniform across the mesh.
        # It must be handled separately from the mixed function space.
        # See https://github.com/scientificcomputing/scifem/blob/main/examples/real_function_space.py
        FS_R = create_real_functionspace(self.mesh_data.mesh)

        FS_op = FS.sub(self._field_idx["op"])
        FS_VEC2, dof_op = FS_op.collapse()
        FS_u = FS.sub(self._field_idx["u"])
        FS_VEC, dof_u = FS_u.collapse()
        FS_ge = FS.sub(self._field_idx["ge"])
        FS_SCAL, dof_ge = FS_ge.collapse()
        FS_gh = FS.sub(self._field_idx["gh"])
        _, dof_gh = FS_gh.collapse()
        FS_phi = FS.sub(self._field_idx["phi"])
        _, dof_phi = FS_phi.collapse()
        FS_T = FS.sub(self._field_idx["T"])
        _, dof_T = FS_T.collapse()
        self.sub_function_spaces = {
            "op": (FS_op, FS_VEC2), 
            "u": (FS_u, FS_VEC), 
            "ge": (FS_ge, FS_SCAL), 
            "gh": (FS_gh, FS_SCAL), 
            "phi": (FS_phi, FS_SCAL), 
            "T": (FS_T, FS_SCAL), 
            "j0": FS_R
        }
        self._sub_dof_maps = {
            "op": dof_op, 
            "u": dof_u, 
            "ge": dof_ge, 
            "gh": dof_gh, 
            "phi": dof_phi, 
            "T": dof_T
        }

        self.field = fem.Function(FS, name="fields", dtype=default_real_type) # Function object stores the function space
        subfs_ufl = ufl.split(self.field) # Tuple containing ufl representations of sub-functions for defining variational forms
        self._sub_fields_tup = self.field.split()
        j0 = fem.Function(FS_R, name="j0", dtype=default_real_type) # Helper field for average boundary current density
        self.sub_fields_ufl = {"j0": j0}
        self.sub_fields = {"j0": j0}
        for name, idx in self._field_idx.items():
            self.sub_fields_ufl[name] = subfs_ufl[idx]
            self.sub_fields[name] = self._sub_fields_tup[idx]
        self.field_pre = fem.Function(FS) # For storing previous time-step fields
        # No need to save dx and ds as class attributes because the call operator will return a new, reconstructed Measure object each time.

    def create_problem(
            self, 
            intrinsic_f: Callable[[Any, Any, Any], Any], 
            trans_strn: Callable[[Any], Any], 
            charge_gap: Callable[[Any], Any], 
            gap_center: Callable[[Any], Any] = lambda op: 0.0
        ):
        """Generate the finite element problem.

        :param intrinsic_f: Callable returning the intrinsic free energy density, with a calling signature ``intrinsic_f(T, op, dop)``, 
            where ``op`` is treated as a 1D array and ``dop`` is the gradient of ``op`` treated as a 2D array with a
            shape ``(op_dim, mesh_dim)``.
        :type intrinsic_f: Callable[[Any, Any, Any], Any]
        :param trans_strn: Callable returning the eigenstrain in Voigt notation, with a calling signature ``trans_strn(op)``.
        :type trans_strn: Callable[[Any], Any]
        :param charge_gap: Callable returning the energy gap, with a calling signature ``charge_gap(op)``.
        :type charge_gap: Callable[[Any], Any]
        :param gap_center: Callable returning the center of the gap measured from a fixed energy level (reference level), 
            with a calling signature ``gap_center(op)``. The default is returning zero.
        :type gap_center: Callable[[Any], Any], optional
        :returns: None

        ----
        Note
        ----
        ``ge`` = (mu_e - ``charge_gap(op)`` / 2 - ``gap_center(op)`` + e ``phi``) / ``T``, 
        ``gh`` = (mu_h - ``charge_gap(op)`` / 2 + ``gap_center(op)`` - e ``phi``) / ``T``, 
        where mu_e and mu_h are the electron and hole quasi-chemical potentials, respectively. The chemical potential and the
        electronic energy (including the electrostatic energy) share the same reference level E_ref which will cancel out. 
        The expressions of ``ge`` and ``gh`` imply that ``phi`` = 0 corresponds to the intrinsic situation free of electrostatic influence,
        i.e., the system is moved infinitely far away from any charges. This means that the reference point of ``phi`` itself can
        be considered to be the infinitly far point away from the system (the most common choice in electrodynamics), regardless 
        of the choice of E_ref. Hence the grounded boundary condition corresponds to ``phi`` = 0 as usual.
        """
        # self.fint = intr_f
        # self.Eg = charge_gap
        # self.e0 = trans_strn

        dx = ufl.Measure("dx", domain=self.mesh_data.mesh, subdomain_data=self.mesh_data.cell_tags, metadata={"quadrature_degree": self.opts["quadr_deg"]})
        ds = ufl.Measure("ds", domain=self.mesh_data.mesh, subdomain_data=self.mesh_data.facet_tags, metadata={"quadrature_degree": self.opts["quadr_deg"]})
        self._dt = fem.Constant(self.mesh_data.mesh, default_real_type(1e-5)) # Time-step size placeholder
        self._noise = fem.Function(self.sub_function_spaces["op"][1], name="noise", dtype=default_real_type) # Noise placeholder

        # op, u, ge, gh, phi, T, j0 = self.sub_fields_ufl
        _op, _u, _ge, _gh, _phi, _T = ufl.split(self.field_pre) # Previous time-step fields
        v = ufl.TestFunctions(self.field.function_space)
        v_op, v_u, v_ge, v_gh, v_phi, v_T = v
        v_j0 = ufl.TestFunction(self.sub_function_spaces["j0"])

        op = ufl.variable(self.sub_fields_ufl["op"])
        dop = ufl.variable(ufl.grad(op))
        f_in = intrinsic_f(self.sub_fields_ufl["T"], op, dop)
        e0 = ufl.as_vector(trans_strn(op))
        Eg = charge_gap(op)
        E0 = gap_center(op)
        Ee = Eg / 2.0 + E0
        Eh = Eg / 2.0 - E0

        mesh_dim = self.mesh_data.mesh.topology.dim
        stiff = self.params["stiffness"]
        if len(stiff) == 2:
            stiff = utils.young_poisson2stiffness(*stiff, dim=mesh_dim)
        stiff = ufl.as_matrix(stiff)
        therm_expan = self.params["therm_expan_coeff"]
        if isinstance(therm_expan, (int, float)):
            therm_expan = [therm_expan] * mesh_dim + [0.0] * (mesh_dim * (mesh_dim - 1) // 2)
        therm_expan = ufl.as_vector(therm_expan)
        strain = utils.ufl_mat2voigt4strain(ufl.sym(ufl.grad(self.sub_fields_ufl["u"])))
        e_therm = therm_expan * (self.sub_fields_ufl["T"] - self.params["therm_expan_Tref"])
        e_elast = strain - e0 - e_therm
        stress = stiff * e_elast # This is matrix-vector multiplication

        m_h2 = 0.332420142 # m / h^2 in unit of eV^-1 nm^-2
        me = self.params["e_eff_mass"]
        mh = self.params["h_eff_mass"]
        Nc = 2.0 * (2.0 * np.pi * m_h2 * me * self.sub_fields_ufl["T"]) ** 1.5
        Nv = 2.0 * (2.0 * np.pi * m_h2 * mh * self.sub_fields_ufl["T"]) ** 1.5
        n = Nc * utils.f1_2(self.sub_fields_ufl["ge"])
        p = Nv * utils.f1_2(self.sub_fields_ufl["gh"])
        _Nc = 2.0 * (2.0 * np.pi * m_h2 * me * _T) ** 1.5
        _Nv = 2.0 * (2.0 * np.pi * m_h2 * mh * _T) ** 1.5
        _n = _Nc * utils.f1_2(_ge)
        _p = _Nv * utils.f1_2(_gh)
        # This is intrinsic chemical potential valid only for a large energy gap compared to T 
        # (but valid for all values of gap if mh = me).
        # TODO: Find a better approximation for the charge-neutral chemical potential for all values
        # of the energy gap.
        mu_neutr = E0 + 3.0 / 4.0 * self.sub_fields_ufl["T"] * np.log(mh / me)
        n_in = Nc * utils.f1_2((mu_neutr - Ee) / self.sub_fields_ufl["T"])
        # Local equilibrium electron and hole densities for given order parameter, potential, and temperature
        n_eq = Nc * utils.f1_2((mu_neutr - Ee + self.sub_fields_ufl["phi"]) / self.sub_fields_ufl["T"])
        p_eq = Nv * utils.f1_2((-mu_neutr - Eh - self.sub_fields_ufl["phi"]) / self.sub_fields_ufl["T"])

        f_in_elast = f_in + ufl.inner(stress, e_elast) / 2.0
        df_dop = ufl.diff(f_in_elast, op) + ufl.diff(Ee, op) * (n - n_in) + ufl.diff(Eh, op) * (p - n_in) 
        df_ddop = ufl.diff(f_in_elast, dop)

        op_rate = self.params["op_relax_rate"]
        if isinstance(op_rate, Iterable):
            op_rate = ufl.as_matrix(op_rate)
        
        mob_e = self.params["e_mobility"]
        if isinstance(mob_e, Iterable):
            mob_e = ufl.as_matrix(mob_e)
        mob_h = self.params["h_mobility"]
        if isinstance(mob_h, Iterable):
            mob_h = ufl.as_matrix(mob_h)
        je = -n * mob_e * ufl.grad(self.sub_fields_ufl["ge"] * self.sub_fields_ufl["T"] + Ee - self.sub_fields_ufl["phi"])
        jh = -p * mob_h * ufl.grad(self.sub_fields_ufl["gh"] * self.sub_fields_ufl["T"] + Eh + self.sub_fields_ufl["phi"])
        j = jh - je
        gen_rad = self.params["eh_recomb_rate"] * (n_eq * p_eq - n * p)

        perm = self.params["back_diel_const"]
        if isinstance(perm, Iterable):
            perm = ufl.as_matrix(perm)
        perm = perm * 0.05526349406 # e V^-1 nm^-1

        # Approximate internal energy relative to that of high-temperature disordered phase
        delU = intrinsic_f(0.0, op, ufl.zero(*dop.ufl_shape))
        _delU = intrinsic_f(0.0, _op, ufl.zero(*dop.ufl_shape))
        joule_heat = ufl.inner(j, ufl.inv(n * mob_e + p * mob_h) * j)
        therm_cond = self.params["therm_conduct"]
        if isinstance(therm_cond, Iterable):
            therm_cond = ufl.as_matrix(therm_cond)
        
        # --- Weak form contributions ---

        # Noise term is w[v_i] = \int w(x) v_i(x) dx, where v_i(x) is a basis test function. 
        # w[v_i] is an uncorrelated set of Gaussian random variables with zero mean and variance 2 T op_rate / dt 
        F_op = ( ufl.inner(op - _op + self._dt * op_rate * df_dop - ufl.sqrt(self._dt) * self._noise, v_op) * dx 
                + self._dt * ufl.inner(df_ddop, ufl.grad(v_op)) * dx )
        F_u = ufl.inner(stress, utils.ufl_mat2voigt4strain(ufl.sym(ufl.grad(v_u)))) * dx
        F_ge = (n - _n - self._dt * gen_rad) * v_ge * dx - self._dt * ufl.inner(je, ufl.grad(v_ge)) * dx
        F_gh = (p - _p - self._dt * gen_rad) * v_gh * dx - self._dt * ufl.inner(jh, ufl.grad(v_gh)) * dx
        F_phi = ufl.inner(perm * ufl.grad(self.sub_fields_ufl["phi"]), ufl.grad(v_phi)) * dx - (p - n) * v_phi * dx
        F_T = ( (self.params["vol_heat_cap"] * (self.sub_fields_ufl["T"] - _T) + delU - _delU - self._dt * joule_heat) * v_T * dx 
               + self._dt * ufl.inner(therm_cond * ufl.grad(self.sub_fields_ufl["T"]), ufl.grad(v_T)) * dx )
        F_field = F_op + F_u + F_ge + F_gh + F_phi + F_T

        # Add natural boundary conditions
        for i, tag, bc in self._bcs_natural:
            F_field += self._dt * bc * v[i] * ds(tag)

        facet_norm = ufl.FacetNormal(self.mesh_data.mesh)
        F_j0 = (self.sub_fields_ufl["j0"] - ufl.dot(j, facet_norm)) * v_j0 * ds(0)

        F = [F_field, F_j0]  # Blocked form

        self.opts["petsc"]["ksp_error_if_not_converged"] = True # Force error if linear solver not converged
        self.opts["petsc"]["snes_error_if_not_converged"] = False # Force pass if nonlinear solver not converged; its convergence will be checked separately.
        self._problem = NonlinearProblem(F, [self.field, self.sub_fields["j0"]], petsc_options_prefix="imt_", bcs=self._bcs_dirichlet, 
                                         kind="mpi", petsc_options=self.opts["petsc"])
        
        # Define von Mises stress
        if mesh_dim == 1:
            stress_vM = stress[0]
        elif mesh_dim == 2:
            stress_vM = ufl.sqrt(stress[0] * stress[0] - stress[0] * stress[1] + stress[1] * stress[1] + 3.0 * stress[2] * stress[2])
        else: # mesh_dim == 3
            stress_vM = ufl.sqrt(0.5 * ( (stress[0] - stress[1]) * (stress[0] - stress[1]) 
                                        + (stress[1] - stress[2]) * (stress[1] - stress[2]) 
                                        + (stress[2] - stress[0]) * (stress[2] - stress[0]) ) 
                                 + 3.0 * (stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]))
        self._exprs2save = {
            # DOLFINx now supports saving vector functions with arbitrary dimension; 
            # see https://fenicsproject.discourse.group/t/saving-vector-function-with-length-greater-than-the-geometrical-dimension/14752
            "op": "op",
            "u": "u",
            "vMs": (stress_vM, self.sub_function_spaces["phi"][1]),
            "n": (n, self.sub_function_spaces["ge"][1]),
            "p": (p, self.sub_function_spaces["gh"][1]),
            "phi": "phi",
            "T": "T"
        }
        vol = self.mesh_data.mesh.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(self.mesh_data.mesh, default_real_type(1.0)) * dx)), op=MPI.SUM)
        s0 = self.mesh_data.mesh.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(self.mesh_data.mesh, default_real_type(1.0)) * ds(0))), op=MPI.SUM)
        self._monitor = {
            "eop_avg": fem.form(ufl.sqrt(op[0] * op[0] + op[1] * op[1] + op[2] * op[2] + op[3] * op[3]) / vol * dx),
            "sop_avg": fem.form(ufl.sqrt(op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7]) / vol * dx),
            "Eg_avg": fem.form(Eg / vol * dx),
            "vMs_avg": fem.form(stress_vM / vol * dx),
            "T_avg": fem.form(self.sub_fields_ufl["T"] / vol * dx),
            "phi0_avg": fem.form(self.sub_fields_ufl["phi"] / s0 * ds(0)),
            "j0_avg": fem.form(self.sub_fields_ufl["j0"] / s0 * ds(0))
        }



        



        

        


    