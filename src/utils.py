from collections.abc import Callable, Sequence
from typing import Optional, Any
from mpi4py import MPI

import numpy as np
from numpy.random import Generator, default_rng

import gmsh  # type: ignore

from petsc4py import PETSc

import ufl
from ufl.core.expr import Expr
from ufl.core.terminal import FormArgument
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio

# Copied from https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_gmsh.html and modified.
def create_mesh(
        comm: MPI.Comm, 
        model: gmsh.model | str, 
        mesh_dim: int = 3, 
        rank: int = 0, 
        mesh_name: str = "mesh", 
        filename: str | None = None, 
        mode: str = "w"
    ) -> gmshio.MeshData:
    """Create a DOLFINx mesh from a Gmsh model and output to file.

    :param comm: MPI communicator top create the mesh on.
    :type comm: MPI.Comm
    :param model: Gmsh model or name of a file containing the mesh (``.msh`` or ``.xdmf``).
    :type model: gmsh.model | str
    :param mesh_dim: Geometric dimension of the mesh of the gmsh model or ``.msh`` file.
    :type mesh_dim: int
    :param rank: Rank of the MPI process (used for generating from gmsh model or reading from ``.msh`` files).
    :type rank: int
    :param mesh_name: Name (identifier) of the mesh to read from the input ``.xdmf`` file and to add to the output file.
    :type mesh_name: str
    :param filename: XDMF filename for writing. Default to ``None`` for not writing to XDMF file.
    :type filename: str | None
    :param mode: XDMF file mode. "w" (write) or "a" (append).
    :type mode: str
    :returns: The created :class:`dolfinx.io.gmsh.MeshData`.
    :rtype: gmshio.MeshData
    """
    if isinstance(model, gmsh.model):
        mesh_data = gmshio.model_to_mesh(model, comm, rank=rank, gdim=mesh_dim)
    elif isinstance(model, str) and model.endswith(".msh"):
        mesh_data = gmshio.read_from_msh(model, comm, rank=rank, gdim=mesh_dim)
    elif isinstance(model, str) and model.endswith(".xdmf"):
        with XDMFFile(comm, model, "r") as in_file:
            mesh = in_file.read_mesh(name=mesh_name)
            try:
                cell_tags = in_file.read_meshtags(mesh, name=f"{mesh_name}_cells")
            except RuntimeError:
                print(f"[create_mesh] no cell tags with identifier '{mesh_name}_cells' found in the xdmf file.")
                cell_tags = None
            try:
                facet_tags = in_file.read_meshtags(mesh, name=f"{mesh_name}_facets")
            except RuntimeError:
                print(f"[create_mesh] no facet tags with identifier '{mesh_name}_facets' found in the xdmf file.")
                facet_tags = None
            try:
                ridge_tags = in_file.read_meshtags(mesh, name=f"{mesh_name}_ridges")
            except RuntimeError:
                print(f"[create_mesh] no ridge tags with identifier '{mesh_name}_ridges' found in the xdmf file.")
                ridge_tags = None
            try:
                peak_tags = in_file.read_meshtags(mesh, name=f"{mesh_name}_peaks")
            except RuntimeError:
                print(f"[create_mesh] no peak tags with identifier '{mesh_name}_peaks' found in the xdmf file.")
                peak_tags = None
        mesh_data = gmshio.MeshData(
            mesh=mesh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            ridge_tags=ridge_tags,
            peak_tags=peak_tags,
            physical_groups={}
        )
    else:
        raise ValueError("[create_mesh] model must be a gmsh.model or a filename ending with .msh or .xdmf")

    mesh_data.mesh.name = mesh_name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{mesh_name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{mesh_name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{mesh_name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{mesh_name}_peaks"
    for i in range(mesh_dim):
        mesh_data.mesh.topology.create_connectivity(mesh_dim - i - 1, mesh_dim)

    if filename is not None:
        with XDMFFile(mesh_data.mesh.comm, filename, mode) as out_file:
            out_file.write_mesh(mesh_data.mesh)
            if mesh_data.cell_tags is not None:
                out_file.write_meshtags(
                    mesh_data.cell_tags,
                    mesh_data.mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh_name}']/Geometry",
                )
            if mesh_data.facet_tags is not None:
                out_file.write_meshtags(
                    mesh_data.facet_tags,
                    mesh_data.mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh_name}']/Geometry",
                )
            if mesh_data.ridge_tags is not None:
                out_file.write_meshtags(
                    mesh_data.ridge_tags,
                    mesh_data.mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh_name}']/Geometry",
                )
            if mesh_data.peak_tags is not None:
                out_file.write_meshtags(
                    mesh_data.peak_tags,
                    mesh_data.mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh_name}']/Geometry",
                )
    return mesh_data

def f1_2(x: Any, exp: Callable = ufl.exp) -> Any:
    """Approximate analytic expression for Fermi-Dirac integral of order 1/2.
    
    :param x: Input value.
    :type x: Any
    :param exp: Exponential function to use. Default to UFL exponential.
    :type exp: Callable
    :returns: Approximated Fermi-Dirac integral of order 1/2 at ``x``.
    :rtype: Any
    """
    sqrt_pi = 1.77245385091
    return 1.0 / (exp(-x) + 3.0 * sqrt_pi / 4.0 * (4.0 + x * x) ** (-0.75))

def ufl_mat2voigt4strain(eps: Expr) -> Expr:
    """Convert a UFL strain matrix to Voigt notation (a UFL vector).
    
    :param eps: Input strain matrix.
    :type eps: Expr
    :returns: Strain in Voigt notation.
    :rtype: Expr
    """
    if eps.ufl_shape == (1, 1):
        return ufl.as_vector([eps[0, 0]])
    elif eps.ufl_shape == (2, 2):
        return ufl.as_vector([
            eps[0, 0],
            eps[1, 1],
            2.0 * eps[0, 1]
        ])
    elif eps.ufl_shape == (3, 3):
        return ufl.as_vector([
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            2.0 * eps[1, 2],
            2.0 * eps[0, 2],
            2.0 * eps[0, 1]
        ])
    else:
        raise ValueError("[ufl_mat2voigt4strain] Input strain matrix must be of shape (1,1), (2,2), or (3,3).")
    
def young_poisson2stiffness(E: float, nu: float, dim: int) -> list[list[float]]:
    """Convert Young's modulus and Poisson's ratio to stiffness tensor in Voigt notation.
    
    :param E: Young's modulus.
    :type E: float
    :param nu: Poisson's ratio.
    :type nu: float
    :param dim: Dimension (1, 2, or 3).
    :type dim: int
    :returns: Stiffness tensor in Voigt notation.
    :rtype: list[list[float]]
    """
    if dim == 1:
        return [[E]]
    elif dim == 2:
        factor = E / (1.0 - nu * nu)
        return [
            [factor,        factor * nu,      0.0],
            [factor * nu,   factor,           0.0],
            [0.0,           0.0,              factor * (1.0 - nu) / 2.0]
        ]
    elif dim == 3:
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return [
            [factor * (1.0 - nu),    factor * nu,           factor * nu,           0.0,                0.0,                0.0],
            [factor * nu,            factor * (1.0 - nu),   factor * nu,           0.0,                0.0,                0.0],
            [factor * nu,            factor * nu,           factor * (1.0 - nu),   0.0,                0.0,                0.0],
            [0.0,                    0.0,                   0.0,                   factor * (1.0 - 2.0 * nu) / 2.0,   0.0,                0.0],
            [0.0,                    0.0,                   0.0,                   0.0,                factor * (1.0 - 2.0 * nu) / 2.0,   0.0],
            [0.0,                    0.0,                   0.0,                   0.0,                0.0,                factor * (1.0 - 2.0 * nu) / 2.0]
        ]
    else:
        raise ValueError("[young_poisson2stiffness] Dimension must be 1, 2, or 3.")

def relativeL2error(
        u1: fem.Function | Sequence[fem.Function], 
        u2: fem.Function | Sequence[fem.Function], 
        eps: float = 1e-10
    ) -> float:
    """Calculate and return the relative L2 error between two :class:`dolfinx.fem.Function`.
    
    :param u1: First DOLFINx Function or sequence of its sub-functions (views). Also used as the reference for computing the relative error.
    :type u1: fem.Function | Sequence[fem.Function]
    :param u2: Second DOLFINx Function or sequence of its sub-functions (views). Must have the same type as ``u1``.
    :type u2: fem.Function | Sequence[fem.Function]
    :param eps: Small value to avoid division by zero.
    :type eps: float
    :returns: Relative L2 error between ``u1`` and ``u2``.
    :rtype: float
    """
    err = 0.0
    if isinstance(u1, Sequence) and isinstance(u2, Sequence):
        for i, u in enumerate(u1):
            du = u - u2[i]
            l2_diff = u.function_space.mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(du, du) * ufl.dx)), op=MPI.SUM)
            l2_u1 = u.function_space.mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(u, u) * ufl.dx)), op=MPI.SUM)
            err += l2_diff / (l2_u1 + eps)
        n_subspaces = len(u1)
    elif isinstance(u1, fem.Function) and isinstance(u2, fem.Function):
        n_subspaces = u1.function_space.num_sub_spaces
        for i in range(n_subspaces):
            u1i = u1.sub(i)
            du = u1i - u2.sub(i)
            l2_diff = u1.function_space.mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(du, du) * ufl.dx)), op=MPI.SUM)
            l2_u1 = u1.function_space.mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(u1i, u1i) * ufl.dx)), op=MPI.SUM)
            err += l2_diff / (l2_u1 + eps)
    else:
        raise TypeError("[relativeL2error] u1 and u2 must be both fem.Function or both tuple of fem.Function.")
    return np.sqrt(err / n_subspaces)

# Copied from http://jsdokken.com/FEniCS23-tutorial/src/approximations.html
class Projector:
    """
    Projector for a given function.
    Solves Ax=b, where

    .. highlight:: python
    .. code-block:: python

        u, v = ufl.TrialFunction(Space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", metadata=metadata)
        A = inner(u, v) * dx
        b = inner(function, v) * dx(metadata=metadata)

    :param function: UFL expression of function to project
    :param space: Space to project function into
    :param petsc_options: Options to pass to PETSc
    :param jit_options: Options to pass to just in time compiler
    :param form_compiler_options: Options to pass to the form compiler
    :param metadata: Data to pass to the integration measure
    """

    _A: PETSc.Mat  # The mass matrix
    _b: PETSc.Vec  # The rhs vector
    _lhs: fem.Form  # The compiled form for the mass matrix
    _ksp: PETSc.KSP  # The PETSc solver
    _x: fem.Function  # The solution vector
    _dx: ufl.Measure  # Integration measure

    def __init__(
        self,
        space: fem.FunctionSpace,
        petsc_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        form_compiler_options: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        petsc_options = {} if petsc_options is None else petsc_options
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options

        # Assemble projection matrix once
        u = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)
        self._dx = ufl.Measure("dx", domain=space.mesh, metadata=metadata)
        a = ufl.inner(u, v) * self._dx(metadata=metadata)
        self._lhs = fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._A = assemble_matrix(self._lhs)
        self._A.assemble()

        # Create vectors to store right hand side and the solution
        self._x = fem.Function(space)
        self._b = fem.Function(space)

        # Create Krylov Subspace solver
        self._ksp = PETSc.KSP().create(space.mesh.comm)
        self._ksp.setOperators(self._A)

        # Set PETSc options
        prefix = f"projector_{id(self)}"
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()
        for opt in opts.getAll().keys():
            del opts[opt]

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(prefix)
        self._A.setFromOptions()
        self._b.x.petsc_vec.setOptionsPrefix(prefix)
        self._b.x.petsc_vec.setFromOptions()

    def reassemble_lhs(self):
        assemble_matrix(self._A, self._lhs)
        self._A.assemble()

    def assemble_rhs(self, h: Expr):
        """
        Assemble the right hand side of the problem
        """
        v = ufl.TestFunction(self._b.function_space)
        rhs = ufl.inner(h, v) * self._dx
        rhs_compiled = fem.form(rhs)
        self._b.x.array[:] = 0.0
        assemble_vector(self._b.x.petsc_vec, rhs_compiled)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def project(self, h: Expr) -> fem.Function:
        """
        Compute projection using a PETSc KSP solver
        """
        self.assemble_rhs(h)
        self._ksp.solve(self._b.x.petsc_vec, self._x.x.petsc_vec)
        return self._x

    def __del__(self):
        self._A.destroy()
        self._ksp.destroy()

class TMat:
    """Class to assemble and store temperature matrix for a given function space and to provide Cholesky 
    decomposition for calculating properly correlated noise out of an uncorrelated noise. The temperature
    matrix T_ij is defined as: ∫ T(x) v_i(x) v_j(x) dx, where T(x) is the temperature field and v_i(x), v_j(x) are
    the basis functions of the function space.

    Currently, parallel backward solving for Cholesky factorization through PETSc is only supported by MKL CPARDISO solver.

    :var lumpedT: The lumped temperature matrix (a vector).
    :vartype lumpedT: PETSc.Vec
    :var lumpedM: The lumped mass matrix (a vector).
    :vartype lumpedM: PETSc.Vec
    """
    _formT: fem.Form  # The compiled form for the temperature matrix
    _form_lumpedT: fem.Form  # The compiled form for the lumped temperature matrix
    _pcT: PETSc.PC # The PETSc preconditioner for the temperature matrix
    _kspM: PETSc.KSP # The PETSc KSP solver for the mass matrix
    _T: PETSc.Mat  # The temperature matrix
    lumpedT: PETSc.Vec # The lumped T matrix (a vector)
    _M: PETSc.Mat # The mass matrix of the function space of the temperature matrix
    lumpedM: PETSc.Vec # The lumped mass matrix (a vector)

    def __init__(self):
        self._formT = None
        self._form_lumpedT = None
        self._pcT = None
        self._kspM = None
        self._T = None
        self.lumpedT = None
        self._M = None
        self.lumpedM = None

    def setT(
            self, 
            T: fem.Function | tuple[FormArgument, fem.FunctionSpace], 
            jit_options: Optional[dict] = None,
            form_compiler_options: Optional[dict] = None,
            metadata: Optional[dict] = None
        ):
        """Attach the temperature field and assemble all the internal data for the first time.

        :param T: Temperature field, represented either as a :class:`dolfinx.fem.Function` or a tuple of its
            UFL representation and (collapsed) function space.
        :type T: fem.Function | tuple[FormArgument, fem.FunctionSpace]
        :param jit_options: Options to pass to just-in-time compiler
        :type jit_options: Optional[dict]
        :param form_compiler_options: Options to pass to the form compiler
        :type form_compiler_options: Optional[dict]
        :param metadata: Data to pass to the integration measure
        :type metadata: Optional[dict]
        """
        if isinstance(T, tuple):
            T_ = T[0]
            V_ = T[1]
        else:
            T_ = T
            V_ = T.function_space

        u = ufl.TrialFunction(V_)
        v = ufl.TestFunction(V_)
        dx = ufl.Measure("dx", domain=V_.mesh, metadata=metadata)
        self._formT = fem.form(T_ * u * v * dx, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._T = assemble_matrix(self._formT)
        self._T.assemble()  # This finalizes the matrix assembly (including update ghosts)
        self._pcT = PETSc.PC().create(V_.mesh.comm)
        self._pcT.setOperators(self._T)

        self._form_lumpedT = fem.form(T_ * v * dx, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self.lumpedT = assemble_vector(self._form_lumpedT)  # This does not finalize the vector assembly (not update ghost yet)
        self.lumpedT.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.lumpedT.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        self._M = assemble_matrix(fem.form(u * v * dx, jit_options=jit_options, form_compiler_options=form_compiler_options))
        self._M.assemble()
        self._kspM = PETSc.KSP().create(V_.mesh.comm)
        self._kspM.setOperators(self._M)
        pc = self._kspM.getPC()
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")

        self.lumpedM = assemble_vector(fem.form(v * dx, jit_options=jit_options, form_compiler_options=form_compiler_options))
        self.lumpedM.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.lumpedM.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        
    def factorT(self, type: str):
        """Assemble and factor the temperature matrix for the attached temperature field with the current value.

        :param type: Type of factorization. E.g., "cholesky".
        :type type: str
        """
        assemble_matrix(self._T, self._formT)
        self._T.assemble()
        self._pcT.setType(type)
        self._pcT.setUp() # ensure factorization is formed

    def get_factorT(self) -> PETSc.Mat:
        """Get the factor matrix of the temperature matrix.
        
        :returns: The factor matrix of the temperature matrix.
        :rtype: PETSc.Mat
        """
        return self._pcT.getFactorMatrix()
    
    def solve_backwardT(self, b: PETSc.Vec, x: PETSc.Vec):
        """Given the factored temperature matrix T = L U, solve the system U x = b.

        :param b: Right-hand side vector.
        :type b: PETSc.Vec
        :param x: Solution vector.
        :type x: PETSc.Vec
        """
        L = self._pcT.getFactorMatrix()
        L.solveBackward(b, x)

    def mat_solveM(self, B: PETSc.Mat, X: PETSc.Mat):
        """Solve the system M X = B, where M is the mass matrix.

        :param B: Right-hand side matrix.
        :type B: PETSc.Mat
        :param X: Solution matrix.
        :type X: PETSc.Mat
        """
        self._kspM.matSolve(B, X)

    def assemble_lumpedT(self):
        """Assemble the lumped temperature matrix :attr:`TMat.lumpedT` for the attached temperature field with the current value."""
        # Zeroing is important before re-assembling
        # self.lumpedT.zeroEntries()  # This does not affect ghost entries
        with self.lumpedT.localForm() as local_vec: # Local form includes ghost entries
            local_vec.zeroEntries()
        assemble_vector(self.lumpedT, self._form_lumpedT)
        self.lumpedT.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.lumpedT.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def __del__(self):
        if self._pcT is not None:
            self._pcT.destroy()
        if self._kspM is not None:
            self._kspM.destroy()
        if self._T is not None:
            self._T.destroy()
        if self.lumpedT is not None:
            self.lumpedT.destroy()
        if self._M is not None:
            self._M.destroy()
        if self.lumpedM is not None:
            self.lumpedM.destroy()

def gen_therm_noise(
        Tmat: TMat,
        dissipUmat: Any,
        rng: Generator,
        noise: fem.Function
    ):
    """Generate thermal noise based on the fluctuation-dissipation theorem.
    It is calculated as: M w = √2 A η L^T, where M is the mass matrix and A, L are the Cholesky factors of the temperature
    and dissipation matrices, respectively. η is a matrix of shape ``(Tmat_dim, dissipUmat_dim)`` containing
    uncorrelated standard normal random numbers, and w of the same shape is the *DOFs* of the proper thermal noise function.
    
    .. note::
        Currently only use method of lumped temperature and mass matrices for efficiency.

    :param Tmat: Assembled :class:`TMat` (temperature matrix) object.
    :type Tmat: TMat
    :param dissipUmat: Upper-triangular Cholesky factor of the dissipation matrix. Can be a scalar for isotropic dissipation.
    :type dissipUmat: Any
    :param rng: Numpy random number generator.
    :type rng: Generator
    :param noise: Output noise field. It is the genuine thermal noise multiplied by √dt so that it is independent of dt, 
        where dt is the time step size.
    :type noise: fem.Function
    """
    mass_dim = noise.function_space.dofmap.index_map.size_local  # Not including ghost dofs
    field_dim = noise.function_space.dofmap.bs
    eta = rng.standard_normal((mass_dim, field_dim))  # The order of indices here matches the DOLFINx Function x array layout

    w = np.dot(eta, dissipUmat)  # numpy.dot works for both scalar and matrix dissip_mat
    w *= (np.sqrt(2.0 * Tmat.lumpedT.array_r) / Tmat.lumpedM.array_r)[:, np.newaxis]  # PETSc.Vec.array_r gives the local (non-ghost) dofs

    # Note reshaping order here; for DOLFINx x array, the field (block) dimension is the fastest varying index 
    # Also note the ghost dofs of DOLFINx Function, which are at the end of the array
    noise.x.array[:(mass_dim * field_dim)] = w.reshape(-1)
    noise.x.scatter_forward()

    

