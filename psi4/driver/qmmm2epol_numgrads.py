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
from psi4.driver.qmmm2epol_gradients import PolarizationHamiltonianGradients


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

        # analytical gradient on QM atoms
        func_DERIV_QM = getattr(polham_grad, function_name + "_DERIV_QM")

        grad_QM = [None for k in range(0, 3*molecule.natom())]
        for i in range(0, molecule.natom()):
            for xyz in [0,1,2]:
                grad_QM[3*i+xyz] = func_DERIV_QM(i, xyz)

        # numerical gradient on QM atoms
        step = 0.001

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


        polarizable_atoms = self.args[3]

        # analytical gradient on polarizable sites
        func_DERIV_POL = getattr(polham_grad, function_name + "_DERIV_POL")

        grad_POL = [None for k in range(0, 3*polarizable_atoms.natom())]
        for i in range(0, polarizable_atoms.natom()):
            for xyz in [0,1,2]:
                print(f"gradient on {xyz}-th coordinate of polarizable atom {i}")
                grad_POL[3*i+xyz] = func_DERIV_POL(i, xyz)

        # numerical gradient on polarizable sites
        step = 0.001

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


        # analytical gradient on point charges

        point_charges = self.args[4]
        func_DERIV_CHG = getattr(polham_grad, function_name + "_DERIV_CHG")

        grad_CHG = [None for k in range(0, 3*point_charges.natom())]
        for i in range(0, point_charges.natom()):
            for xyz in [0,1,2]:
                print(f"gradient on {xyz}-th coordinate of point charge {i}")
                grad_CHG[3*i+xyz] = func_DERIV_CHG(i, xyz)

        # numerical gradient on point charges
        step = 0.001

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
        
    def compare_contracted_gradients_F(self, D):
        """
        compare analytical and numerical gradients for the contraction

                            (elec)
            FD    = sum    F      (Rpol(i))  D
                       m,n  m,n               m,n

        for a symmetric density matrix D
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad_QM, grad_POL = polham_grad._contracted_gradients_F(D)

        # numerical gradient on QM atoms
        step = 0.001

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
                FD_plus = np.einsum('imn,mn->i', F_plus, D)
                # f(x-dx)
                F_minus = shift_coords(-step)._polarization_integrals_F()
                FD_minus = np.einsum('imn,mn->i', F_minus, D)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (FD_plus - FD_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of contraction  sum_{m,n} F^(elec)_{i,m,n} D_{m,n}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        step = 0.001

        npol = polarizable_atoms.natom()
        grad_POL_NUM = np.zeros((3*npol, 3))
        
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
                FD_plus = np.einsum('imn,mn->i', F_plus, D)
                # f(x-dx)
                F_minus = shift_coords(-step)._polarization_integrals_F()
                FD_minus = np.einsum('imn,mn->i', F_minus, D)
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                dFD = (FD_plus - FD_minus)/(2*step)
                # Integrals on other polarizable centers j should not change if we move the polarizable center i != j,
                # so all terms dFD[3*j:3*(j+1)] = 0 unless j == i
                grad_POL_NUM[3*i:3*(i+1),xyz] = dFD[3*i:3*(i+1)]

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of contraction  sum_{m,n} F^(elec)_{i,m,n} D_{m,n}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")


        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5

        # all tests passed
        return True

    def compare_contracted_gradients_I(self, D):
        """
        compare analytical and numerical gradients for the contraction

                            (elec)
            ID    = sum    I   [a,b]   (Rpol(i))  D
                       m,n  m,n                    m,n

        for a symmetric density matrix D
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad_QM, grad_POL = polham_grad._contracted_gradients_I(D)

        # numerical gradient on QM atoms
        step = 0.001

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
                ID_plus = np.einsum('iabmn,mn->iab', I_plus, D)
                # f(x-dx)
                I_minus = shift_coords(-step)._polarization_integrals_I()
                ID_minus = np.einsum('iabmn,mn->iab', I_minus, D)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (ID_plus - ID_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of contraction  sum_{m,n} I^(elec)_{i,a,b;m,n} D_{m,n}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        step = 0.001

        npol = polarizable_atoms.natom()
        grad_POL_NUM = np.zeros((3*npol, 3,3))
        
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
                ID_plus = np.einsum('iabmn,mn->iab', I_plus, D)
                # f(x-dx)
                I_minus = shift_coords(-step)._polarization_integrals_I()
                ID_minus = np.einsum('iabmn,mn->iab', I_minus, D)
                
                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                dID = (ID_plus - ID_minus)/(2*step)
                # Integrals on other polarizable centers j should not change if we move the polarizable center i != j,
                # so all terms dID[j,:,:] = 0 unless j == i
                grad_POL_NUM[3*i+xyz,:,:] = dID[i,:,:]

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of contraction  sum_{m,n} I^(elec)_{i,a,b;m,n} D_{m,n}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")


        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5

        # all tests passed
        return True

    def compare_coulomb_J_gradients(self, D1, D2):
        """
        compare analytical and numerical gradients for the contraction

          dJ                       (1)  d (pq|h^(2)|rs)   (2)
          --(D1,D2) =  sum        D     ---------------  D
          dx              p,q,r,s  p,q        d x         r,s

        for symmetric density matrices D1 and D2
        """
        polham_grad = PolarizationHamiltonianGradients(*self.args, **self.kwds)
        
        molecule, basis, ribasis, polarizable_atoms = self.args[0:4]

        # analytical gradient on QM atoms and point charges
        grad_QM, grad_POL = polham_grad.coulomb_J_GRAD(D1, D2)

        # numerical gradient on QM atoms
        step = 0.001

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
                J_plus = shift_coords(step).coulomb_J(D2)
                DJD_plus = np.einsum('mn,mn', D1, J_plus)
                # f(x-dx)
                J_minus = shift_coords(-step).coulomb_J(D2)
                DJD_minus = np.einsum('mn,mn', D1, J_minus)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_QM_NUM[3*i+xyz] = (DJD_plus - DJD_minus)/(2*step)

        grad_QM = np.array(grad_QM)
        grad_QM_NUM = np.array(grad_QM_NUM)
        err_QM = la.norm(grad_QM - grad_QM_NUM)

        print("+++ Gradient of J-like contraction  sum_{p,q,r,s} D1_{p,q} d(pq|h^(2)|rs)/dx D2_{r,s}   on QM atoms +++")
        print(" Analytical:")
        print(grad_QM)
        print(" Numerical:")
        print(grad_QM_NUM)
        print(f" Error: {err_QM}")

        # numerical gradient on polarizable sites
        step = 0.001

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
                J_plus = shift_coords(step).coulomb_J(D2)
                DJD_plus = np.einsum('mn,mn', D1, J_plus)
                # f(x-dx)
                J_minus = shift_coords(-step).coulomb_J(D2)
                DJD_minus = np.einsum('mn,mn', D1, J_minus)

                # df/dx = (f(x+dx) - f(x-dx))/(2 dx)
                grad_POL_NUM[3*i+xyz] = (DJD_plus - DJD_minus)/(2*step)

        grad_POL = np.array(grad_POL)
        err_POL = la.norm(grad_POL - grad_POL_NUM)

        print("+++ Gradient of J-like contraction  sum_{p,q,r,s} D1_{p,q} d(pq|h^(2)|rs)/dx D1_{r,s}   on polarizable sites +++")
        print(" Analytical:")
        print(grad_POL)
        print(" Numerical:")
        print(grad_POL_NUM)
        print(f" Error: {err_POL}")


        assert err_QM  < 1.0e-5
        assert err_POL < 1.0e-5

        # all tests passed
        return True
