KB_E = 8.61733326e-5  # k_B / e in SI units

class VO2:
    def __init__(self):
        v = 0.059  # nm^3
        self.Tc = 338 * KB_E  # eV
        self.T1 = 270.0 * KB_E  # eV
        self.A = 3.943 * self.Tc / v  # eV / nm^3
        self.B11 = 1.368 * self.Tc / v  # eV / nm^3
        self.B13 = -3.679 * self.Tc / v  # eV / nm^3
        self.B12 = 1000.0 * self.Tc / v  # eV / nm^3
        self.B14 = 1000.0 * self.Tc / v  # eV / nm^3
        self.C11 = 0.4 * self.Tc / v  # eV / nm^3
        self.C13 = 2.0 * self.Tc / v  # eV / nm^3
        self.C12 = 100.0 * self.Tc / v  # eV / nm^3
        self.C14 = 100.0 * self.Tc / v  # eV / nm^3

        self.T2 = 275.0 * KB_E  # eV
        self.a = 2.057 * self.Tc / v  # eV / nm^3
        self.b11 = -0.623 * self.Tc / v  # eV / nm^3
        self.b13 = 0.121 * self.Tc / v  # eV / nm^3
        self.b12 = 1000.0 * self.Tc / v  # eV / nm^3
        self.b14 = 1000.0 * self.Tc / v  # eV / nm^3
        self.c11 = 0.331 * self.Tc / v  # eV / nm^3
        self.c13 = 4.189 * self.Tc / v  # eV / nm^3
        self.c12 = 100.0 * self.Tc / v  # eV / nm^3
        self.c14 = 100.0 * self.Tc / v  # eV / nm^3

        self.K = 0.3 * self.Tc / v  # eV / nm^3
        self.K1111 = 0.2 * self.Tc / v  # eV / nm^3
        self.K1133 = -1.5 * self.Tc / v  # eV / nm^3
        self.K1313 = 0.075 * self.Tc / v  # eV / nm^3
        self.K1122 = -1000.0 * self.Tc / v  # eV / nm^3
        self.K1144 = -1000.0 * self.Tc / v  # eV / nm^3
        self.k1111 = 0.05 * self.Tc / v  # eV / nm^3
        self.k1133 = 0.667 * self.Tc / v  # eV / nm^3

        self.G = 1.0  # 0.4; eV / nm
        self.g = 1.0  # 0.16; eV / nm

        H1 = -15.725 * self.Tc / v  # eV / nm^3
        H2 = -18.040 * self.Tc / v  # eV / nm^3
        H3 = -0.063 * self.Tc / v  # eV / nm^3
        H4 = -30.0 * self.Tc / v  # eV / nm^3
        H5 = 20.0 * self.Tc / v  # eV / nm^3
        h1 = -34.913 * self.Tc / v  # eV / nm^3
        h2 = -4.755 * self.Tc / v  # eV / nm^3
        h3 = 0.137 * self.Tc / v  # eV / nm^3
        h4 = 73.392 * self.Tc / v  # eV / nm^3
        h5 = 61.233 * self.Tc / v  # eV / nm^3

        C11 = 110.0 / 1.60217663e-1  # eV / nm^3; = C22
        C33 = 140.0 / 1.60217663e-1
        C12 = 81.6 / 1.60217663e-1
        C13 = 56.5 / 1.60217663e-1  # = C23
        C44 = 44.2 / 1.60217663e-1  # = C55
        C66 = 81.9 / 1.60217663e-1

        self.S1 = (H1 * C13 - H2 * C33) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self.S2 = H3 / (2.0 * (C11 - C12))
        self.S3 = (2.0 * H2 * C13 - H1 * (C11 + C12)) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self.S4 = H5 / (8.0 * C44)
        self.S5 = H4 / (8.0 * C66)
        self.s1 = (h1 * C13 - h2 * C33) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self.s2 = h3 / (2.0 * (C11 - C12))
        self.s3 = (2.0 * h2 * C13 - h1 * (C11 + C12)) / (2.0 * (C33 * (C11 + C12) - 2.0 * C13 * C13))
        self.s4 = h5 / (8.0 * C44)
        self.s5 = h4 / (8.0 * C66)

        self.Eg0 = 0.3 # eV

    def fint(self, T, op, dop):
        """Intrinsic free energy density. `op` has 8 components, with first 4 for electronic order
        parameters and last 4 for structural order parameters.
        """
        f = (
            self.A / 2.0 * (T - self.T1) / self.Tc * (op[0] * op[0] + op[1] * op[1] + op[2] * op[2] + op[3] * op[3])
            + self.B11 / 4.0 * (op[0] * op[0] * op[0] * op[0] + op[1] * op[1] * op[1] * op[1] 
                                + op[2] * op[2] * op[2] * op[2] + op[3] * op[3] * op[3] * op[3])
            + self.B13 / 2.0 * (op[0] * op[0] * op[2] * op[2] + op[1] * op[1] * op[3] * op[3])
            + self.B12 / 2.0 * (op[0] * op[0] * op[1] * op[1] + op[2] * op[2] * op[3] * op[3])
            + self.B14 / 2.0 * (op[0] * op[0] * op[3] * op[3] + op[1] * op[1] * op[2] * op[2])
            + self.C11 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[0] * op[0] + op[1] * op[1] * op[1] * op[1] * op[1] * op[1]
                                + op[2] * op[2] * op[2] * op[2] * op[2] * op[2] + op[3] * op[3] * op[3] * op[3] * op[3] * op[3])
            + self.C13 / 6.0 * (op[0] * op[0] * op[2] * op[2] * op[2] * op[2] + op[2] * op[2] * op[0] * op[0] * op[0] * op[0]
                                + op[1] * op[1] * op[3] * op[3] * op[3] * op[3] + op[3] * op[3] * op[1] * op[1] * op[1] * op[1])
            + self.C12 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[1] * op[1] + op[1] * op[1] * op[1] * op[1] * op[0] * op[0]
                                + op[2] * op[2] * op[2] * op[2] * op[3] * op[3] + op[3] * op[3] * op[3] * op[3] * op[2] * op[2])
            + self.C14 / 6.0 * (op[0] * op[0] * op[0] * op[0] * op[3] * op[3] + op[3] * op[3] * op[3] * op[3] * op[0] * op[0]
                                + op[1] * op[1] * op[1] * op[1] * op[2] * op[2] + op[2] * op[2] * op[2] * op[2] * op[1] * op[1])
            + self.a / 2.0 * (T - self.T2) / self.Tc * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            + self.b11 / 4.0 * (op[4] * op[4] * op[4] * op[4] + op[5] * op[5] * op[5] * op[5] 
                                + op[6] * op[6] * op[6] * op[6] + op[7] * op[7] * op[7] * op[7])
            + self.b13 / 2.0 * (op[4] * op[4] * op[6] * op[6] + op[5] * op[5] * op[7] * op[7])
            + self.b12 / 2.0 * (op[4] * op[4] * op[5] * op[5] + op[6] * op[6] * op[7] * op[7])
            + self.b14 / 2.0 * (op[4] * op[4] * op[7] * op[7] + op[5] * op[5] * op[6] * op[6])
            + self.c11 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[4] * op[4] + op[5] * op[5] * op[5] * op[5] * op[5] * op[5]
                                + op[6] * op[6] * op[6] * op[6] * op[6] * op[6] + op[7] * op[7] * op[7] * op[7] * op[7] * op[7])
            + self.c13 / 6.0 * (op[4] * op[4] * op[6] * op[6] * op[6] * op[6] + op[6] * op[6] * op[4] * op[4] * op[4] * op[4]
                                + op[5] * op[5] * op[7] * op[7] * op[7] * op[7] + op[7] * op[7] * op[5] * op[5] * op[5] * op[5])
            + self.c12 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[5] * op[5] + op[5] * op[5] * op[5] * op[5] * op[4] * op[4]
                                + op[6] * op[6] * op[6] * op[6] * op[7] * op[7] + op[7] * op[7] * op[7] * op[7] * op[6] * op[6])
            + self.c14 / 6.0 * (op[4] * op[4] * op[4] * op[4] * op[7] * op[7] + op[7] * op[7] * op[7] * op[7] * op[4] * op[4]
                                + op[5] * op[5] * op[5] * op[5] * op[6] * op[6] + op[6] * op[6] * op[6] * op[6] * op[5] * op[5])
            + self.K * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            - self.K1111 / 2.0 * (op[0] * op[0] * op[4] * op[4] + op[1] * op[1] * op[5] * op[5]
                                 + op[2] * op[2] * op[6] * op[6] + op[3] * op[3] * op[7] * op[7])
            - self.K1133 / 2.0 * (op[0] * op[0] * op[6] * op[6] + op[1] * op[1] * op[7] * op[7] 
                                  + op[2] * op[2] * op[4] * op[4] + op[3] * op[3] * op[5] * op[5])
            - self.K1313 * 2.0 * (op[0] * op[2] * op[4] * op[6] + op[1] * op[3] * op[5] * op[7])
            - self.K1122 / 2.0 * (op[0] * op[0] * op[5] * op[5] + op[1] * op[1] * op[4] * op[4]
                                  + op[2] * op[2] * op[7] * op[7] + op[3] * op[3] * op[6] * op[6])
            - self.K1144 / 2.0 * (op[0] * op[0] * op[7] * op[7] + op[1] * op[1] * op[6] * op[6]
                                  + op[2] * op[2] * op[5] * op[5] + op[3] * op[3] * op[4] * op[4])
            + self.k1111 / 2.0 * (op[0] * op[4] * op[4] * op[4] + op[1] * op[5] * op[5] * op[5]
                                  + op[2] * op[6] * op[6] * op[6] + op[3] * op[7] * op[7] * op[7])
            + self.k1133 * 3.0 / 2.0 * (op[0] * op[4] * op[6] * op[6] + op[2] * op[4] * op[4] * op[6]
                                        + op[1] * op[5] * op[7] * op[7] + op[3] * op[5] * op[5] * op[7])
        )

        g = ( 
            self.G / 2.0 * (dop[0,0] * dop[0,0] + dop[0,1] * dop[0,1] + dop[0,2] * dop[0,2]
                            + dop[1,0] * dop[1,0] + dop[1,1] * dop[1,1] + dop[1,2] * dop[1,2]
                            + dop[2,0] * dop[2,0] + dop[2,1] * dop[2,1] + dop[2,2] * dop[2,2]
                            + dop[3,0] * dop[3,0] + dop[3,1] * dop[3,1] + dop[3,2] * dop[3,2])
            + self.g / 2.0 * (dop[4,0] * dop[4,0] + dop[4,1] * dop[4,1] + dop[4,2] * dop[4,2]
                              + dop[5,0] * dop[5,0] + dop[5,1] * dop[5,1] + dop[5,2] * dop[5,2]
                              + dop[6,0] * dop[6,0] + dop[6,1] * dop[6,1] + dop[6,2] * dop[6,2]
                              + dop[7,0] * dop[7,0] + dop[7,1] * dop[7,1] + dop[7,2] * dop[7,2])
        )

        return f + g
    
    def e0(self, op):
        """Transformation strain in Voigt notation. `op` has 8 components, with first 4 for electronic order
        parameters and last 4 for structural order parameters.
        """
        e1 = (
            self.S1 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            + self.S2 * (op[0] * op[4] - op[1] * op[5] + op[2] * op[6] - op[3] * op[7])
            + self.s1 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            + self.s2 * (op[4] * op[4] - op[5] * op[5] + op[6] * op[6] - op[7] * op[7])
        )
        e2 = (
            self.S1 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            - self.S2 * (op[0] * op[4] - op[1] * op[5] + op[2] * op[6] - op[3] * op[7])
            + self.s1 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
            - self.s2 * (op[4] * op[4] - op[5] * op[5] + op[6] * op[6] - op[7] * op[7])
        )
        e3 = (
            self.S3 * (op[0] * op[4] + op[1] * op[5] + op[2] * op[6] + op[3] * op[7])
            + self.s3 * (op[4] * op[4] + op[5] * op[5] + op[6] * op[6] + op[7] * op[7])
        )
        e4 = self.S4 / 2.0 * (op[1] * op[7] + op[3] * op[5]) + self.s4 * op[5] * op[7]
        e5 = -self.S4 / 2.0 * (op[0] * op[6] + op[2] * op[4]) - self.s4 * op[4] * op[6]
        e6 = (
            -self.S5 * (op[0] * op[4] + op[1] * op[5] - op[2] * op[6] - op[3] * op[7]) 
            - self.s5 * (op[4] * op[4] + op[5] * op[5] - op[6] * op[6] - op[7] * op[7])
        )
        return [e1, e2, e3, e4, e5, e6]
    
    def Eg(self, op):
        """Charge gap. `op` has 8 components, with first 4 for electronic order parameters.
        """
        return self.Eg0 * (op[0] * op[0] + op[1] * op[1] + op[2] * op[2] + op[3] * op[3])