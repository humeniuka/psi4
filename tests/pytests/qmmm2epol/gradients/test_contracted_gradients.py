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
    check gradients of contractions

      * dA/dx(U)
      * d(diagA)/dx(V)
      * dF^(e)/dx(E)
      * dF^(n)/dx(N)
      * dS/dx(R)

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

def test_coulomb_exchange_gradients():
    """
    check gradients of Coulomb and exchange energy
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

    # random density matrices
    D1 = np.random.rand(nbf, nbf)
    D2 = np.random.rand(nbf, nbf)
    # symmetrize matrices
    D1 = 0.5 * (D1 + np.einsum('mn->nm', D1))
    D2 = 0.5 * (D2 + np.einsum('mn->nm', D2))

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_coulomb_J_gradients(D1, D2)

    # Since the psi4 integral machinery assumes that density matrices are symmetric,
    # we cannot compute dK/dx(D1,D2) for D1 != D2. The reason is that when psi4 evaluates
    # the contraction 
    #   dF/dx(E) = dF_{i;m,n}/dx E_{i;m,n}
    # it assumes that E_{i;m,n} == E{i;n,m} and only sums over one triangle of the density matrix.
    # However, the intermediate matrices that arise during the construction of dK/dx are not 
    # symmetric.
    assert grad_comp.compare_exchange_K_gradients(D1, D1)


if __name__ == "__main__":
    #test_qmmm2epol_gradients()
    #test_contracted_gradients()
    test_coulomb_exchange_gradients()
