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
from psi4.driver.qmmm2epol_contracted_gradients import PolarizationHamiltonianGradients
from psi4.driver.qmmm2epol_contracted_numgrads import GradientComparison
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
    check QM gradients of a contraction of polarization integrals F^(elec) 
    with an array of (constant) density matrices
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

    # number of polarizable sites
    npol = polham_grad.npol
    # number of AOs
    nbf = polham_grad.nbf

    # make random numbers reproducible
    np.random.seed(101)

    # random tensor V for contracting with the gradients of diag(A)
    V = np.random.rand(npol, 3, 3)

    # random matrix D for contracting with the gradients of overlap matrix
    D = np.random.rand(nbf, nbf)
    # symmetrize matrices
    D = 0.5 * (D + np.einsum('mn->nm', D))

    # random tensor E for contracting with the gradients of F^(elec)_{i,m,n}
    E = np.random.rand(3*npol, nbf, nbf)
    # symmetrize matrices
    E = 0.5 * (E + np.einsum('imn->inm', E))

    # random tensor Y for contracting with the gradients of I^(elec)_{k,a,b,m,n}
    Y = np.random.rand(npol, 3, 3, nbf, nbf)
    # symmetrize matrices in 2nd and 3rd indices
    Y = 0.5 * (Y + np.einsum('kabmn->kbamn', Y))
    # symmetrize matrices in 4th and 5th indices
    Y = 0.5 * (Y + np.einsum('kabmn->kabnm', Y))

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_contracted_gradients_diagA(V)
    assert grad_comp.compare_contracted_gradients_overlap(D)

    assert grad_comp.compare_contracted_gradients_F(E)
    assert grad_comp.compare_contracted_gradients_I(Y)


if __name__ == "__main__":
    test_qmmm2epol_gradients()
    test_contracted_gradients()
