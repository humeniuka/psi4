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
from psi4.driver.qmmm2epol_gradients import PolarizationHamiltonianGradients, RHF_QMMM2ePolGradients
from psi4.driver.qmmm2epol_numgrads import GradientComparison, GradientComparisonRHF
from psi4.driver.qmmm2epol_io import amber2psi4

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

    # random tensor E for contracting with the gradients of F^(elec)_{i,m,n}
    E = np.random.rand(3*npol, nbf, nbf)

    # random tensor Y for contracting with the gradients of I^(elec)_{k,a,b,m,n}
    Y = np.random.rand(npol, 3, 3, nbf, nbf)
    # symmetrize matrices in 2nd and 3rd indices
    Y = 0.5 * (Y + np.einsum('kabmn->kbamn', Y))

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_contracted_gradients_diagA(V)
    assert grad_comp.compare_contracted_gradients_overlap(D)

    assert grad_comp.compare_contracted_gradients_F(E)
    assert grad_comp.compare_contracted_gradients_I(Y)

def test_zero_electron_part_gradients():
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

def test_one_electron_part_gradients():
    """
    check gradients of one-electron energy of polarization Hamiltonian
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

    # random density matrix
    D = np.random.rand(nbf, nbf)
#    # symmetrize matrix
#    D = 0.5 * (D + np.einsum('mn->nm', D))

    # 1) All integrals are treated with resolution of identity (R.I.)
    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   same_site_integrals='R.I.',
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_one_electron_part_gradients(D)

    # 2) Same-site integrals are treated exactly.
    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   same_site_integrals='exact',
                                   verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_one_electron_part_gradients(D)

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
#    # symmetrize matrices
#    D1 = 0.5 * (D1 + np.einsum('mn->nm', D1))
#    D2 = 0.5 * (D2 + np.einsum('mn->nm', D2))

    grad_comp = GradientComparison(molecule, basis, ribasis,
                                   polarizable_atoms, point_charges,
                                   polarizabilities=polarizabilities,
                                   verbose=0)

    # compute analytical and numerical gradients and compare them
    assert grad_comp.compare_coulomb_J_gradients(D1, D2)
    assert grad_comp.compare_exchange_K_gradients(D1, D2)

def test_RHF_energy_gradients():
    """
    compare analytical and numerical gradients of the plain RHF energy
    """
    # load QM and MM regions from AMBER files
    molecule, polarizable_atoms, point_charges, polarizabilities = amber2psi4('solvated.prmtop', 'solvated.rst7', 'solvated.qmregion')

    # plain RHF, no polarizable environment of point charges
    polarizable_atoms = psi4.geometry("")
    point_charges = psi4.geometry("")

    # run closed-shell SCF calculation
    grad_comp = GradientComparisonRHF(molecule, polarizable_atoms, point_charges, 'sto-3g',
                                      polarizabilities=polarizabilities,
                                      verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_energy_gradients()


def test_RHF_QMMM2ePol_energy_gradients():
    """
    compare analytical and numerical gradients of the RHF+QM/MM-2e-Pol energy
    """
    # load QM and MM regions from AMBER files
    molecule, polarizable_atoms, point_charges, polarizabilities = amber2psi4('solvated.prmtop', 'solvated.rst7', 'solvated.qmregion')

    # run closed-shell SCF calculation
    grad_comp = GradientComparisonRHF(molecule, polarizable_atoms, point_charges, 'sto-3g',
                                      polarizabilities=polarizabilities,
                                      same_site_integrals="exact", #"R.I.",
                                      verbose=0)

    # compute analytical and numerical QM gradients and compare them
    assert grad_comp.compare_energy_gradients()


if __name__ == "__main__":
    test_contracted_gradients()

    test_zero_electron_part_gradients()
    test_one_electron_part_gradients()
    test_coulomb_exchange_gradients()

    test_RHF_energy_gradients()
    test_RHF_QMMM2ePol_energy_gradients()

