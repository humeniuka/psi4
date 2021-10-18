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
        step = 0.01

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
        step = 0.01

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
        step = 0.01

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
