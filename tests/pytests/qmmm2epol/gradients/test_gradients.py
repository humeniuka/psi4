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
    """
    compare analytical and numerical gradients for the following terms of the polarization Hamiltonian:

      * 0-electron part h^(0)

    """
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

def test_contracted_gradients():
    """
    check QM gradients of a contraction of polarization integrals F^(elec) with a (constant) density matrix 
    """
    # load QM and MM regions from AMBER files
    molecule, polarizable_atoms, point_charges, polarizabilities = amber2psi4('solvated.prmtop', 'solvated.rst7', 'solvated.qmregion')
    
    # basis for QM atoms
    wfn = psi4.core.Wavefunction.build(molecule, 'sto-3g')
    basis = wfn.basisset()
    ribasis = basis

    polham_grad = PolarizationHamiltonianGradients(molecule, basis, ribasis,
                                                   polarizable_atoms, point_charges,
                                                   polarizabilities=polarizabilities,
                                                   verbose=0)

    nbf = polham_grad.nbf

    # make random numbers reproducible
    np.random.seed(101)

    # random matrix D1
    D1 = np.random.rand(nbf, nbf)
    # symmetrize matrix
    D1 = 0.5 * (D1 + D1.T)

    # random matrix D2
    D2 = np.random.rand(nbf, nbf)
    # symmetrize matrix
    D2 = 0.5 * (D2 + D2.T)

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_contracted_gradients_F(D1)
    assert grad_comp.compare_contracted_gradients_I(D1)

    assert grad_comp.compare_coulomb_J_gradients(D1, D2)

    

if __name__ == "__main__":
    test_contracted_gradients()
    test_qmmm2epol_gradients()

