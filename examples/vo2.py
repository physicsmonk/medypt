import sys
sys.path.append('../src')

import numpy as np
from mpi4py import MPI

import gmsh

from dolfinx import default_real_type
from dolfinx.fem import Function, Constant

from imt import IMTModel
from maters import KB_E, VO2


model = IMTModel()

msh = "mesh.msh"  # Or None if generating mesh within this script using gmsh

has_bl = False
lx = 10.0
ly = 5.0
lz = 2.0
lc = 1.0

l_bl1 = 0.1  # thickness of first boundary layer
n_bl = 5     # number of boundary layers
r_bl = 1.5   # growth rate of boundary layer thickness

bdr_tags = {
    "left": 0,
    "right": 1,
    "front": 2,
    "back": 3,
    "bottom": 4,
    "top": 5
}

substr_strain = [0.005, 0.005, 0.0]  # in-plane substrate strain: [e_xx, e_yy, e_xy]

nitsche_eps = 1e-5

mob = 5e4
n_ref = 10.0
h = 3e6 / 1.380649e4  # Heat transfer coefficient; nm^-2 ns^-1
model.params["op_relax_rate"] = 100.0

model.opts["min_dt"] = 1e-12
model.opts["rand_seed"] = 321245
model.opts["verbose"] = True
model.opts["save_period"] = 10
model.opts["has_thermal_noise"] = True

T_i = 300.0 * KB_E
mu_e_i = 0.0
mu_h_i = 0.0
op_i = np.array([-0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.5, 0.0]) 
phi_i = 0.0
tspan = 0.001
dt = 1e-5


l_bc = l_bl1 if has_bl else lc

p_rank = MPI.COMM_WORLD.Get_rank()

if msh is None:
    gmsh.initialize()
    msh = gmsh.model()
    if p_rank == 0:
        msh.add("box")
        msh.setCurrent("box")
        if has_bl:
            # NOTE: Use occ, not geo kernel, to define entities before geo.extrudeBoundaryLayer call, 
            # or we could encounter weird errors, such as "Unable to subdivide extruded mesh"!

            # Define bottom and top surfaces for subsequent extrusion
            p1_gmsh = msh.occ.addPoint(0, 0, 0, lc)
            p2_gmsh = msh.occ.addPoint(lx, 0, 0, lc)
            p3_gmsh = msh.occ.addPoint(lx, ly, 0, lc)
            p4_gmsh = msh.occ.addPoint(0, ly, 0, lc)
            p5_gmsh = msh.occ.addPoint(0, 0, lz, lc)
            p6_gmsh = msh.occ.addPoint(lx, 0, lz, lc)
            p7_gmsh = msh.occ.addPoint(lx, ly, lz, lc)
            p8_gmsh = msh.occ.addPoint(0, ly, lz, lc)

            l1_gmsh = msh.occ.addLine(p1_gmsh, p2_gmsh)
            l2_gmsh = msh.occ.addLine(p2_gmsh, p3_gmsh)
            l3_gmsh = msh.occ.addLine(p3_gmsh, p4_gmsh)
            l4_gmsh = msh.occ.addLine(p4_gmsh, p1_gmsh)
            # Opposite direction compared to above to get the top surface norm pointing inward
            l5_gmsh = msh.occ.addLine(p5_gmsh, p8_gmsh)
            l6_gmsh = msh.occ.addLine(p8_gmsh, p7_gmsh)
            l7_gmsh = msh.occ.addLine(p7_gmsh, p6_gmsh)
            l8_gmsh = msh.occ.addLine(p6_gmsh, p5_gmsh)

            cl5_gmsh = msh.occ.addCurveLoop([l1_gmsh, l2_gmsh, l3_gmsh, l4_gmsh])
            cl6_gmsh = msh.occ.addCurveLoop([l5_gmsh, l6_gmsh, l7_gmsh, l8_gmsh])

            s1_gmsh = msh.occ.addPlaneSurface([cl5_gmsh])  # Bottom surface, norm pointing up
            s2_gmsh = msh.occ.addPlaneSurface([cl6_gmsh])  # Top surface, norm pointing down

            msh.occ.synchronize()

            d_gmsh = [l_bl1] # thickness of first layer
            for i in range(1, n_bl): d_gmsh.append(d_gmsh[-1] + d_gmsh[0] * r_bl**i)
            ext_bl_gmsh = msh.geo.extrudeBoundaryLayer(msh.getEntities(2), [1] * n_bl, d_gmsh)
            msh.geo.synchronize()

            bdr_tl_gmsh = [ext_bl_gmsh[0], ext_bl_gmsh[6]]  # get "top" surfaces of the boundary layer
            bdr_l_gmsh = msh.getBoundary(bdr_tl_gmsh, recursive=False)
            bdr_p_gmsh = msh.getBoundary(bdr_tl_gmsh, recursive=True)

            # These lines connect the two boundary layers
            l11_gmsh = msh.geo.addLine(bdr_p_gmsh[0][1], bdr_p_gmsh[4][1])
            l12_gmsh = msh.geo.addLine(bdr_p_gmsh[1][1], bdr_p_gmsh[-1][1])
            l13_gmsh = msh.geo.addLine(bdr_p_gmsh[2][1], bdr_p_gmsh[-2][1])
            l14_gmsh = msh.geo.addLine(bdr_p_gmsh[3][1], bdr_p_gmsh[-3][1])

            cl11_gmsh = msh.geo.addCurveLoop([bdr_l_gmsh[0][1], l12_gmsh, bdr_l_gmsh[-1][1], -l11_gmsh])
            cl12_gmsh = msh.geo.addCurveLoop([bdr_l_gmsh[1][1], l13_gmsh, bdr_l_gmsh[-2][1], -l12_gmsh])
            cl13_gmsh = msh.geo.addCurveLoop([bdr_l_gmsh[2][1], l14_gmsh, bdr_l_gmsh[-3][1], -l13_gmsh])
            cl14_gmsh = msh.geo.addCurveLoop([bdr_l_gmsh[3][1], l11_gmsh, bdr_l_gmsh[4][1], -l14_gmsh])

            s11_gmsh = msh.geo.addPlaneSurface([cl11_gmsh])
            s12_gmsh = msh.geo.addPlaneSurface([cl12_gmsh])
            s13_gmsh = msh.geo.addPlaneSurface([cl13_gmsh])
            s14_gmsh = msh.geo.addPlaneSurface([cl14_gmsh])

            sl_gmsh = msh.geo.addSurfaceLoop([s11_gmsh, s12_gmsh, s13_gmsh, s14_gmsh, bdr_tl_gmsh[0][1], bdr_tl_gmsh[1][1]])

            v_gmsh = msh.geo.addVolume([sl_gmsh])

            msh.geo.synchronize()

            # Add physical groups
            msh.addPhysicalGroup(3, [v_gmsh, ext_bl_gmsh[1][1], ext_bl_gmsh[7][1]], 1)
            msh.addPhysicalGroup(2, [s11_gmsh, ext_bl_gmsh[2][1], ext_bl_gmsh[-1][1]], bdr_tags["front"])
            msh.addPhysicalGroup(2, [s12_gmsh, ext_bl_gmsh[3][1], ext_bl_gmsh[-2][1]], bdr_tags["right"])
            msh.addPhysicalGroup(2, [s13_gmsh, ext_bl_gmsh[4][1], ext_bl_gmsh[-3][1]], bdr_tags["back"])
            msh.addPhysicalGroup(2, [s14_gmsh, ext_bl_gmsh[5][1], ext_bl_gmsh[-4][1]], bdr_tags["left"])
            msh.addPhysicalGroup(2, [s1_gmsh], bdr_tags["bottom"])
            msh.addPhysicalGroup(2, [s2_gmsh], bdr_tags["top"])
        else:
            v_gmsh = msh.occ.addBox(0.0, 0.0, 0.0, lx, ly, lz)

            msh.occ.synchronize()

            p_gmsh = msh.getEntities(0)
            # print(p_gmsh)
            msh.mesh.setSize(p_gmsh, lc)  # Set mesh size for points

            s_gmsh = msh.getBoundary([(3, v_gmsh)])  # left, right, front, back, bottom, top surfaces
            msh.addPhysicalGroup(3, [v_gmsh], 1)
            msh.addPhysicalGroup(2, [s_gmsh[0][1]], bdr_tags["left"])
            msh.addPhysicalGroup(2, [s_gmsh[1][1]], bdr_tags["right"])
            msh.addPhysicalGroup(2, [s_gmsh[2][1]], bdr_tags["front"])
            msh.addPhysicalGroup(2, [s_gmsh[3][1]], bdr_tags["back"])
            msh.addPhysicalGroup(2, [s_gmsh[4][1]], bdr_tags["bottom"])
            msh.addPhysicalGroup(2, [s_gmsh[5][1]], bdr_tags["top"])

        # Generate the 3D mesh
        msh.mesh.generate(3)
        gmsh.write("mesh.msh")

model.load_mesh(MPI.COMM_WORLD, msh, 3, rank=0)

if isinstance(msh, gmsh.model):
    gmsh.finalize()


vo2 = VO2()

model.load_physics(["op", "T", "u", "eh", "phi"], op_dim=8, intrinsic_f=vo2.intrinsic_f, charge_gap=vo2.charge_gap,
                   trans_strn=vo2.trans_strn, gap_center=lambda op: 0.0)

u_bott = Function(model.fields["u"].function_space)
u_bott.interpolate(lambda x: np.vstack((substr_strain[0] * x[0] + substr_strain[2] * x[1],
                                        substr_strain[2] * x[0] + substr_strain[1] * x[1],
                                        0.0 * x[0])))
volt = Constant(model.mesh_data.mesh, default_real_type(phi_i))
def flux_e(fields):
    return n_ref * mob / (nitsche_eps * l_bc) * (fields["T"] * fields["ge"] + vo2.charge_gap(fields["op"]) / 2.0)
def flux_h(fields):
    return n_ref * mob / (nitsche_eps * l_bc) * (fields["T"] * fields["gh"] + vo2.charge_gap(fields["op"]) / 2.0)
def T_flux(fields):
    return h * (fields["T"] - T_i)
bcs = [
    ("u", bdr_tags["bottom"], u_bott), 
    ("phi", bdr_tags["front"], volt),
    ("ge", bdr_tags["left"], flux_e),
    ("gh", bdr_tags["left"], flux_h),
    ("ge", bdr_tags["right"], flux_e),
    ("gh", bdr_tags["right"], flux_h),
    ("T", bdr_tags["bottom"], T_flux)
]
model.set_bcs(bcs)

model.create_problem()

def op0(x):
    return np.full((8, x.shape[1]), op_i[:, np.newaxis])
def T0(x):
    return np.full(x.shape[1], T_i)
def ge0(x):
    return np.full(x.shape[1], (mu_e_i - vo2.charge_gap(op_i) / 2.0) / T_i)
def gh0(x):
    return np.full(x.shape[1], (mu_h_i - vo2.charge_gap(op_i) / 2.0) / T_i)
def phi0(x):
    return np.full(x.shape[1], phi_i)

ics = {
    "op": op0,
    "u": u_bott,
    "phi": phi0,
    "T": T0,
    "ge": ge0,
    "gh": gh0
}

model.solve(tspan, dt, ics)
