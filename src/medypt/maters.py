"""Classes defining example materials."""

KB_E = 8.61733326e-5  
"""k_B / e in SI units"""

M_H2 = 0.332420142 
"""m / h^2 in unit of eV^-1 nm^-2"""

EPS0 = 0.05526349406 
"""Vacuum permittivity in unit of e V^-1 nm^-1"""

class VO2:
    """Class defining required callables (see :py:meth:`~medypt.model.ModelBase.create_problem`) for the material VO2.
    """
    def __init__(self):
        v = 0.059  # nm^3
        self._Tc = 338 * KB_E  # eV
        self._T1 = 270.0 * KB_E  # eV
        self._A = 3.943 * self._Tc / v  # eV / nm^3
        self._B11 = 1.368 * self._Tc / v  # eV / nm^3
        self._B13 = -3.679 * self._Tc / v  # eV / nm^3
        self._B12 = 1000.0 * self._Tc / v  # eV / nm^3
        self._B14 = 1000.0 * self._Tc / v  # eV / nm^3
        self._C11 = 0.4 * self._Tc / v  # eV / nm^3
        self._C13 = 2.0 * self._Tc / v  # eV / nm^3
        self._C12 = 100.0 * self._Tc / v  # eV / nm^3
        self._C14 = 100.0 * self._Tc / v  # eV / nm^3

        self._T2 = 275.0 * KB_E  # eV
        self._a = 2.057 * self._Tc / v  # eV / nm^3
        self._b11 = -0.623 * self._Tc / v  # eV / nm^3
        self._b13 = 0.121 * self._Tc / v  # eV / nm^3
        self._b12 = 1000.0 * self._Tc / v  # eV / nm^3
        self._b14 = 1000.0 * self._Tc / v  # eV / nm^3
        self._c11 = 0.331 * self._Tc / v  # eV / nm^3
        self._c13 = 4.189 * self._Tc / v  # eV / nm^3
        self._c12 = 100.0 * self._Tc / v  # eV / nm^3
        self._c14 = 100.0 * self._Tc / v  # eV / nm^3

        self._K = 0.3 * self._Tc / v  # eV / nm^3
        self._K1111 = 0.2 * self._Tc / v  # eV / nm^3
        self._K1133 = -1.5 * self._Tc / v  # eV / nm^3
        self._K1313 = 0.075 * self._Tc / v  # eV / nm^3
        self._K1122 = -1000.0 * self._Tc / v  # eV / nm^3
        self._K1144 = -1000.0 * self._Tc / v  # eV / nm^3
        self._k1111 = 0.05 * self._Tc / v  # eV / nm^3
        self._k1133 = 0.667 * self._Tc / v  # eV / nm^3

        self._G = 1.0  # 0.4; eV / nm
        self._g = 1.0  # 0.16; eV / nm

        H1 = -15.725 * self._Tc / v  # eV / nm^3
        H2 = -18.040 * self._Tc / v  # eV / nm^3
        H3 = -0.063 * self._Tc / v  # eV / nm^3
        H4 = -30.0 * self._Tc / v  # eV / nm^3
        H5 = 20.0 * self._Tc / v  # eV / nm^3
        h1 = -34.913 * self._Tc / v  # eV / nm^3
        h2 = -4.755 * self._Tc / v  # eV / nm^3
        h3 = 0.137 * self._Tc / v  # eV / nm^3
        h4 = 73.392 * self._Tc / v  # eV / nm^3
        h5 = 61.233 * self._Tc / v  # eV / nm^3

        C11 = 110.0 / 1.60217663e-1  # eV / nm^3; = C22
        C33 = 140.0 / 1.60217663e-1
        C12 = 81.6 / 1.60217663e-1
        C13 = 56.5 / 1.60217663e-1  # = C23
        C44 = 44.2 / 1.60217663e-1  # = C55
        C66 = 81.9 / 1.60217663e-1

        self._S1 = (H1 * C13 - H2 * C33) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self._S2 = H3 / (2.0 * (C11 - C12))
        self._S3 = (2.0 * H2 * C13 - H1 * (C11 + C12)) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self._S4 = H5 / (8.0 * C44)
        self._S5 = H4 / (8.0 * C66)
        self._s1 = (h1 * C13 - h2 * C33) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self._s2 = h3 / (2.0 * (C11 - C12))
        self._s3 = (2.0 * h2 * C13 - h1 * (C11 + C12)) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self._s4 = h5 / (8.0 * C44)
        self._s5 = h4 / (8.0 * C66)

        self._Eg0 = 0.6 # eV

    def intrinsic_f(self, T, op, dop):
        """Intrinsic free energy density.

        :param T: Temperature in eV.
        :param op: Order parameters (dimensionless), treated as an 1D array with 8 components. First 4 are electronic order parameters,
            last 4 are structural order parameters.
        :param dop: Gradient of order parameters in nm^-1, treated as a 2D array with shape (8, 3).
        :returns: Free energy density in eV / nm^3.
        """
        f = (
            self._A / 2.0 * (T - self._T1) / self._Tc * (op[0] * op[0] + op[1] * op[1] + op[2] * op[2] + op[3] * op[3])
            + self._B11 / 4.0 * (op[0] * op[0] * op[0] * op[0] + op[1] * op[1] * op[1] * op[1] 
                                + op[2] * op[2] * op[2] * op[2] + op[3] * op[3] * op[3] * op[3])
            + self._B13 / 2.0 * (op[0] * op[0] * op[2] * op[2] + op[1] * op[1] * op[3] * op[3])
            + self._B12 / 2.0 * (op[0] * op[0] * op[1] * op[1] + op[2] * op[2] * op[3] * op[3])
            + self._B14 / 2.0 * (op[0] * op[0] * op[3] * op[3] + op[1] * op[1] * op[2] * op[2])
            + self._C11 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[0] * op[0] + op[1] * op[1] * op[1] * op[1] * op[1] * op[1]
                                + op[2] * op[2] * op[2] * op[2] * op[2] * op[2] + op[3] * op[3] * op[3] * op[3] * op[3] * op[3])
            + self._C13 / 6.0 * (op[0] * op[0] * op[2] * op[2] * op[2] * op[2] + op[2] * op[2] * op[0] * op[0] * op[0] * op[0]
                                + op[1] * op[1] * op[3] * op[3] * op[3] * op[3] + op[3] * op[3] * op[1] * op[1] * op[1] * op[1])
            + self._C12 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[1] * op[1] + op[1] * op[1] * op[1] * op[1] * op[0] * op[0]
                                + op[2] * op[2] * op[2] * op[2] * op[3] * op[3] + op[3] * op[3] * op[3] * op[3] * op[2] * op[2])
            + self._C14 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[3] * op[3] + op[3] * op[3] * op[3] * op[3] * op[0] * op[0]
                                + op[1] * op[1] * op[1] * op[1] * op[2] * op[2] + op[2] * op[2] * op[2] * op[2] * op[1] * op[1])
            + self._a / 2.0 * (T - self._T2) / self._Tc * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            + self._b11 / 4.0 * (op[4] * op[4] * op[4] * op[4] + op[5] * op[5] * op[5] * op[5] 
                                + op[6] * op[6] * op[6] * op[6] + op[7] * op[7] * op[7] * op[7])
            + self._b13 / 2.0 * (op[4] * op[4] * op[6] * op[6] + op[5] * op[5] * op[7] * op[7])
            + self._b12 / 2.0 * (op[4] * op[4] * op[5] * op[5] + op[6] * op[6] * op[7] * op[7])
            + self._b14 / 2.0 * (op[4] * op[4] * op[7] * op[7] + op[5] * op[5] * op[6] * op[6])
            + self._c11 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[4] * op[4] + op[5] * op[5] * op[5] * op[5] * op[5] * op[5]
                                + op[6] * op[6] * op[6] * op[6] * op[6] * op[6] + op[7] * op[7] * op[7] * op[7] * op[7] * op[7])
            + self._c13 / 6.0 * (op[4] * op[4] * op[6] * op[6] * op[6] * op[6] + op[6] * op[6] * op[4] * op[4] * op[4] * op[4]
                                + op[5] * op[5] * op[7] * op[7] * op[7] * op[7] + op[7] * op[7] * op[5] * op[5] * op[5] * op[5])
            + self._c12 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[5] * op[5] + op[5] * op[5] * op[5] * op[5] * op[4] * op[4]
                                + op[6] * op[6] * op[6] * op[6] * op[7] * op[7] + op[7] * op[7] * op[7] * op[7] * op[6] * op[6])
            + self._c14 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[7] * op[7] + op[7] * op[7] * op[7] * op[7] * op[4] * op[4]
                                + op[5] * op[5] * op[5] * op[5] * op[6] * op[6] + op[6] * op[6] * op[6] * op[6] * op[5] * op[5])
            + self._K * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            - self._K1111 / 2.0 * (op[0] * op[0] * op[4] * op[4] + op[1] * op[1] * op[5] * op[5]
                                 + op[2] * op[2] * op[6] * op[6] + op[3] * op[3] * op[7] * op[7])
            - self._K1133 / 2.0 * (op[0] * op[0] * op[6] * op[6] + op[1] * op[1] * op[7] * op[7] 
                                  + op[2] * op[2] * op[4] * op[4] + op[3] * op[3] * op[5] * op[5])
            - self._K1313 * 2.0 * (op[0] * op[2] * op[4] * op[6] + op[1] * op[3] * op[5] * op[7])
            - self._K1122 / 2.0 * (op[0] * op[0] * op[5] * op[5] + op[1] * op[1] * op[4] * op[4]
                                  + op[2] * op[2] * op[7] * op[7] + op[3] * op[3] * op[6] * op[6])
            - self._K1144 / 2.0 * (op[0] * op[0] * op[7] * op[7] + op[1] * op[1] * op[6] * op[6]
                                  + op[2] * op[2] * op[5] * op[5] + op[3] * op[3] * op[4] * op[4])
            + self._k1111 / 2.0 * (op[0] * op[4] * op[4] * op[4] + op[1] * op[5] * op[5] * op[5]
                                  + op[2] * op[6] * op[6] * op[6] + op[3] * op[7] * op[7] * op[7])
            + self._k1133 * 3.0 / 2.0 * (op[0] * op[4] * op[6] * op[6] + op[2] * op[4] * op[4] * op[6]
                                        + op[1] * op[5] * op[7] * op[7] + op[3] * op[5] * op[5] * op[7])
        )

        g = ( 
            self._G / 2.0 * (dop[0,0] * dop[0,0] + dop[0,1] * dop[0,1] + dop[0,2] * dop[0,2]
                            + dop[1,0] * dop[1,0] + dop[1,1] * dop[1,1] + dop[1,2] * dop[1,2]
                            + dop[2,0] * dop[2,0] + dop[2,1] * dop[2,1] + dop[2,2] * dop[2,2]
                            + dop[3,0] * dop[3,0] + dop[3,1] * dop[3,1] + dop[3,2] * dop[3,2])
            + self._g / 2.0 * (dop[4,0] * dop[4,0] + dop[4,1] * dop[4,1] + dop[4,2] * dop[4,2]
                              + dop[5,0] * dop[5,0] + dop[5,1] * dop[5,1] + dop[5,2] * dop[5,2]
                              + dop[6,0] * dop[6,0] + dop[6,1] * dop[6,1] + dop[6,2] * dop[6,2]
                              + dop[7,0] * dop[7,0] + dop[7,1] * dop[7,1] + dop[7,2] * dop[7,2])
        )

        return f + g
    
    def trans_strn(self, op):
        """Transformation strain in Voigt notation.

        :param op: Order parameters (dimensionless), treated as an 1D array with 8 components. First 4 are electronic order parameters,
            last 4 are structural order parameters.
        :returns: Transformation strain in Voigt notation as a list with 6 components.
        """
        e1 = (
            self._S1 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            + self._S2 * (op[0] * op[4] - op[1] * op[5] + op[2] * op[6] - op[3] * op[7])
            + self._s1 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            + self._s2 * (op[4] * op[4] - op[5] * op[5] + op[6] * op[6] - op[7] * op[7])
        )
        e2 = (
            self._S1 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            - self._S2 * (op[0] * op[4] - op[1] * op[5] + op[2] * op[6] - op[3] * op[7])
            + self._s1 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            - self._s2 * (op[4] * op[4] - op[5] * op[5] + op[6] * op[6] - op[7] * op[7])
        )
        e3 = (
            self._S3 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            + self._s3 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
        )
        e4 = self._S4 / 2.0 * (op[1] * op[7] + op[3] * op[5]) + self._s4 * op[5] * op[7]
        e5 = -self._S4 / 2.0 * (op[0] * op[6] + op[2] * op[4]) - self._s4 * op[4] * op[6]
        e6 = (
            -self._S5 * (op[0] * op[4] + op[1] * op[5] - op[2] * op[6] - op[3] * op[7]) 
            - self._s5 * (op[4] * op[4] + op[5] * op[5] - op[6] * op[6] - op[7] * op[7])
        )
        return [e1, e2, e3, e4, e5, e6]
    
    def charge_gap(self, op):
        """Charge gap.

        :param op: Order parameters (dimensionless), treated as an 1D array with 8 components. First 4 are electronic order parameters,
            last 4 are structural order parameters.
        :returns: Charge gap in eV.
        """
        return self._Eg0 * (op[0] * op[0] + op[1] * op[1] + op[2] * op[2] + op[3] * op[3])