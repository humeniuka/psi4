#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2021 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

"""
Helper class for computing numerical gradients of the polarization Hamiltonian (for debugging purposes only)
"""

import numpy as np
import numpy.linalg as la
import psi4

from psi4.driver.qmmm2epol import PolarizationHamiltonian
from psi4.driver.qmmm2epol_contracted_gradients import PolarizationHamiltonianGradients


class GradientComparison:
    def __init__(self, *args, **kwds):
        """
        The positional arguments and keywords are the same as those of the constructor of 
        the classes `PolarizationHamiltonian` and `PolarizationHamiltonianGradient`
        """
        self.args = args
        self.kwds = kwds
    def compare_gradients(self, function_name):
        """
        compare analytical and numerical gradients for the quantity computed by the member
        function `function_name`.

        The function computes the gradients
         * on QM atoms
         * polarizable sites and
         * point charges
        first analytically and then by a two-point finite difference formula.

        Example
        -------
        >>> ...
        >>> G = GradientComparison(molecule, basis, ribasis, polarizable_atoms, point_charges)
        >>> G.compare_gradients('zero_electron_part')

        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule = self.args[0]

        # analytical gradients
        func_GRAD = getattr(polham_grad, function_name + "_GRAD")
        grad = func_GRAD()
        grad_QM, grad_POL, grad_CHG = polham_grad.split_gradient(grad)
        
        # Step size for finite differences
        step = 0.001

        # numerical gradient on QM atoms
        grad_QM_NUM = [None for k in range(0, 3*molecule.natom())]
        for i in range(0, molecule.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    molecule_ = molecule.clone()
                    coords = molecule_.geometry().np
                    coords[i,xyz] += step
                    molecule_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[0] = molecule_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                # call member function by its name
                h_plus = getattr(shift_coords(step), function_name)()
                # f(x-dx)
                h_minus = getattr(shift_coords(-step), function_name)()

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (h_plus - h_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print(f"+++ Gradient of {function_name} on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        polarizable_atoms = self.args[3]

        grad_POL_NUM = [None for k in range(0, 3*polarizable_atoms.natom())]
        for i in range(0, polarizable_atoms.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    polarizable_atoms_ = polarizable_atoms.clone()
                    coords = polarizable_atoms_.geometry().np
                    coords[i,xyz] += step
                    polarizable_atoms_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[3] = polarizable_atoms_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                # call member function by its name
                h_plus = getattr(shift_coords(step), function_name)()

                # f(x-dx)
                h_minus = getattr(shift_coords(-step), function_name)()
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_POL_NUM[3*i+xyz] = (h_plus - h_minus)/(2*step)

        grad_POL = np.array(grad_POL)
        grad_POL_NUM = np.array(grad_POL_NUM)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print(f"+++ Gradient of {function_name} on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")

        # numerical gradient on point charges
        point_charges = self.args[4]

        grad_CHG_NUM = [None for k in range(0, 3*point_charges.natom())]
        for i in range(0, point_charges.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    point_charges_ = point_charges.clone()
                    coords = point_charges_.geometry().np
                    coords[i,xyz] += step
                    point_charges_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[4] = point_charges_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                # call member function by its name
                h_plus = getattr(shift_coords(step), function_name)()

                # f(x-dx)
                h_minus = getattr(shift_coords(-step), function_name)()
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_CHG_NUM[3*i+xyz] = (h_plus - h_minus)/(2*step)

        grad_CHG = np.array(grad_CHG)
        grad_CHG_NUM = np.array(grad_CHG_NUM)
        err_CHG = la.norm(grad_CHG - grad_CHG_NUM)

        print(f"+++ Gradient of {function_name} on point charges +++")
        print(" Analytical:")
        print(grad_CHG)
        print(" Numerical:")
        print(grad_CHG_NUM)
        print(f" Error: {err_CHG}")

        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5
        assert err_CHG < 1.0e-5

        # all tests passed
        return True

    def compare_contracted_gradients_F(self, E):
        """
        compare analytical and numerical gradients for the contraction

          dF                        (elec)
          -- (E)  = sum      d/dx (F      )   E
          dx           i,m,n        i;m,n      i;m,n

        assuming that all E[i,:,:] are symmetric matrices.
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad = polham_grad._gradFelec(E)
        grad_QM, grad_POL, grad_CHG = polham_grad.split_gradient(grad)

        assert la.norm(grad_CHG) == 0.0

        # step size for finite differences
        step = 0.001

        # numerical gradient on QM atoms
        grad_QM_NUM = [None for k in range(0, 3*molecule.natom())]
        for i in range(0, molecule.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    molecule_ = molecule.clone()
                    coords = molecule_.geometry().np
                    coords[i,xyz] += step
                    molecule_.set_geometry(psi4.core.Matrix.from_array(coords))

                    # The centers of the basis functions are taken from the basis object.
                    wfn = psi4.core.Wavefunction.build(molecule_, basis.name() )
                    basis_ = wfn.basisset()

                    args_ = list(self.args[:])
                    args_[0:3] = (molecule_, basis_, basis_)
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                F_plus = shift_coords(step)._polarization_integrals_F()
                FE_plus = np.einsum('imn,imn', F_plus, E)
                # f(x-dx)
                F_minus = shift_coords(-step)._polarization_integrals_F()
                FE_minus = np.einsum('imn,imn', F_minus, E)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (FE_plus - FE_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of contraction  sum_{i,m,n} F^(elec)_{i,m,n} E_{i,m,n}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        npol = polarizable_atoms.natom()
        grad_POL_NUM = np.zeros(3*npol)
        
        for i in range(0, npol):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    polarizable_atoms_ = polarizable_atoms.clone()
                    coords = polarizable_atoms_.geometry().np
                    coords[i,xyz] += step
                    polarizable_atoms_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[3] = polarizable_atoms_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                F_plus = shift_coords(step)._polarization_integrals_F()
                FE_plus = np.einsum('imn,imn', F_plus, E)
                # f(x-dx)
                F_minus = shift_coords(-step)._polarization_integrals_F()
                FE_minus = np.einsum('imn,imn', F_minus, E)
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_POL_NUM[3*i+xyz] = (FE_plus - FE_minus)/(2*step)

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of contraction  sum_{i,m,n} F^(elec)_{i,m,n} E_{i,m,n}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")


        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5

        # all tests passed
        return True

    def compare_contracted_gradients_I(self, Y):
        """
        compare analytical and numerical gradients for the contraction

          dI                              (elec)
          -- (Y)  = sum            d/dx (I         )   Y
          dx           k,a,b,m,n          k;a,b;m,n      k;a,b;m,n

        assuming that all Y[k,a,b,:,:] are symmetric matrices.
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad = polham_grad._gradI(Y)
        grad_QM, grad_POL, grad_CHG = polham_grad.split_gradient(grad)

        assert la.norm(grad_CHG) == 0.0

        # step size for finite differences
        step = 0.001

        # numerical gradient on QM atoms
        grad_QM_NUM = [None for k in range(0, 3*molecule.natom())]
        for i in range(0, molecule.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    molecule_ = molecule.clone()
                    coords = molecule_.geometry().np
                    coords[i,xyz] += step
                    molecule_.set_geometry(psi4.core.Matrix.from_array(coords))

                    # The centers of the basis functions are taken from the basis object.
                    wfn = psi4.core.Wavefunction.build(molecule_, basis.name() )
                    basis_ = wfn.basisset()

                    args_ = list(self.args[:])
                    args_[0:3] = (molecule_, basis_, basis_)
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                I_plus = shift_coords(step)._polarization_integrals_I()
                IY_plus = np.einsum('kabmn,kabmn', I_plus, Y)
                # f(x-dx)
                I_minus = shift_coords(-step)._polarization_integrals_I()
                IY_minus = np.einsum('kabmn,kabmn', I_minus, Y)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (IY_plus - IY_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of contraction  sum_{k,a,b,m,n} I^(elec)_{k,a,b,m,n} Y_{k,a,b,m,n}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        npol = polarizable_atoms.natom()
        grad_POL_NUM = np.zeros(3*npol)
        
        for i in range(0, npol):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    polarizable_atoms_ = polarizable_atoms.clone()
                    coords = polarizable_atoms_.geometry().np
                    coords[i,xyz] += step
                    polarizable_atoms_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[3] = polarizable_atoms_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                # f(x+dx)
                I_plus = shift_coords(step)._polarization_integrals_I()
                IY_plus = np.einsum('kabmn,kabmn', I_plus, Y)
                # f(x-dx)
                I_minus = shift_coords(-step)._polarization_integrals_I()
                IY_minus = np.einsum('kabmn,kabmn', I_minus, Y)
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_POL_NUM[3*i+xyz] = (IY_plus - IY_minus)/(2*step)

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of contraction  sum_{k,a,b,m,n} I^(elec)_{k,a,b,m,n} Y_{k,a,b,m,n}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")


        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5

        # all tests passed
        return True

    def compare_contracted_gradients_overlap(self, R):
        """
        compare analytical and numerical gradients for the contraction of the
        overlap matrix with some density matrix R

          dS 
          -- (R)  = sum    d/dx (S   ) R
          dx           m,n        m,n   m,n

        assuming that R is symmetric.
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad = polham_grad._gradS(R)
        grad_QM, grad_POL, grad_CHG = polham_grad.split_gradient(grad)

        assert la.norm(grad_CHG) == 0.0
        assert la.norm(grad_POL) == 0.0

        # step size for finite differences
        step = 0.001

        # numerical gradient on QM atoms
        grad_QM_NUM = [None for k in range(0, 3*molecule.natom())]
        for i in range(0, molecule.natom()):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    molecule_ = molecule.clone()
                    coords = molecule_.geometry().np
                    coords[i,xyz] += step
                    molecule_.set_geometry(psi4.core.Matrix.from_array(coords))

                    # The centers of the basis functions are taken from the basis object.
                    wfn = psi4.core.Wavefunction.build(molecule_, basis.name() )
                    basis_ = wfn.basisset()

                    # compute the overlap matrix
                    mints_ri = psi4.core.MintsHelper(basis_)
                    S = mints_ri.ao_overlap()
                    return S
                # f(x+dx)
                S_plus = shift_coords(step)
                SR_plus = np.einsum('mn,mn', S_plus, R)
                # f(x-dx)
                S_minus = shift_coords(-step)
                SR_minus = np.einsum('mn,mn', S_minus, R)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (SR_plus - SR_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of contraction  sum_{m,n} S_{m,n} R_{m,n}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        assert err_QM  < 1.0e-5

        # all tests passed
        return True

    def compare_contracted_gradients_diagA(self, V):
        """
        compare analytical and numerical gradients for the contraction

           d diag(A)                    d A_{3*k+a,3*k+b}
           --------- (V) = sum  sum     ----------------- V
             d x              k    a,b        d x          k,a,b

        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad = polham_grad._gradDiagA(V)
        grad_QM, grad_POL, grad_CHG = polham_grad.split_gradient(grad)

        assert la.norm(grad_QM ) == 0.0
        assert la.norm(grad_CHG) == 0.0

        # step size for finite differences
        step = 0.001

        # numerical gradient on polarizable sites
        npol = polarizable_atoms.natom()
        grad_POL_NUM = np.zeros(3*npol)
        
        for i in range(0, npol):
            for xyz in [0,1,2]:
                def shift_coords(step):
                    polarizable_atoms_ = polarizable_atoms.clone()
                    coords = polarizable_atoms_.geometry().np
                    coords[i,xyz] += step
                    polarizable_atoms_.set_geometry(psi4.core.Matrix.from_array(coords))
                    args_ = list(self.args[:])
                    args_[3] = polarizable_atoms_
                    polham = PolarizationHamiltonian(*args_, **self.kwds)
                    return polham
                def contract_diagonal(A, V):
                    AV = 0.0
                    npol = polham_grad.npol
                    for k in range(0, npol):
                        for a in [0,1,2]:
                            for b in [0,1,2]:
                                AV += A[3*k+a,3*k+b] * V[k,a,b]
                    return AV
                # f(x+dx)
                A_plus = shift_coords(step).A
                AV_plus = contract_diagonal(A_plus, V)
                # f(x-dx)
                A_minus = shift_coords(-step).A
                AV_minus = contract_diagonal(A_minus, V)
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_POL_NUM[3*i+xyz] = (AV_plus - AV_minus)/(2*step)

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of contraction  sum_{k,a,b} diag(A)_{k,a,b} V_{k,a,b}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")

        assert err_POL < 1.0e-5

        # all tests passed
        return True
