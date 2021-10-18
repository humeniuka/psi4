#!/usr/bin/env python
"""
compare analytical and numerical gradients of different parts of the polarization Hamiltonian
"""
import pytest
pytestmark = pytest.mark.quick

import numpy as np
import numpy.linalg as la

import psi4

from psi4.driver.qmmm2epol import PolarizationHamiltonian
from psi4.driver.qmmm2epol_gradients import PolarizationHamiltonianGradients
from psi4.driver.qmmm2epol_numgrads import GradientComparison
from psi4.driver.qmmm2epol_io import amber2psi4

@pytest.mark.smoke
def test_qmmm2epol_gradients():
    # load QM and MM regions from AMBER files
    molecule, polarizable_atoms, point_charges, polarizabilities = amber2psi4('solvated.prmtop', 'solvated.rst7', 'solvated.qmregion')
    
    # basis for QM atoms
    wfn = psi4.core.Wavefunction.build(molecule, 'sto-3g')
    basis = wfn.basisset()
    ribasis = basis

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    assert grad_comp.compare_gradients('zero_electron_part')

if __name__ == "__main__":
    test_qmmm2epol_gradients()

