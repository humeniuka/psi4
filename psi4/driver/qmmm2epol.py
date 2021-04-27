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
Module with classes to integrate polarizable atoms into a QM calculation
"""
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import psi4
from psi4 import core

# atomic dipole polarizabilities in bohr^3
_theoretical_polarizabilities = {
    # He: theoretical value at FCI/cc-pVDZ level of theory
    "HE" :  0.30274713
}

def _T(rvec):
    """
    dipole polarization tensor (a 3x3 matrix)

                  delta_ab      3
    T  (rvec)  =  --------  -  --- rvec[a] rvec[b] 
     ab             r^3        r^5

    Parameters
    ----------
    rvec   :    np.ndarray
      3-dimensional vector

    Returns
    -------
    T      :    np.ndarray
      3x3 matrix
    """
    T = np.zeros((3,3))
    
    r = la.norm(rvec)
    r3 = pow(r,3)
    r5 = pow(r,5)
    
    for a in range(0,3):
        T[a,a] = 1.0/r3
        for b in range(a,3):
            T[a,b] -= 3.0/r5 * rvec[a]*rvec[b]
            T[b,a]  = T[a,b]

    return T

class PolarizationHamiltonian(object):
    def __init__(self, molecule, basis, ribasis, polarizable_atoms, point_charges,
                 polarizabilities=_theoretical_polarizabilities):
        """
        two-electron, one-electron and zero-electron contributions to electronic
        Hamiltonian due to the presence of polarizable sites

        Parameters
        ----------
        molecule           :  psi4.core.Molecule
          QM part
        basis              :  psi4.core.BasisSet
          basis set for QM part
        ribasis            :  psi4.core.BasisSet
          auxiliary basis set for QM part (used for resolution of identity)
        polarizable_atoms  :  psi4.core.Molecule
          polarizable atoms in MM part
        point_charges      :  psi4.core.Molecule
          MM atoms which carry point charges
        polarizabilities   :  dict
          dictionary with atomic polarizabilities for each atom type
        """
        self.molecule = molecule
        self.basis = basis
        # auxiliary basis set for resolution of identity
        self.ribasis = ribasis
        
        self.polarizable_atoms = polarizable_atoms
        assert self.polarizable_atoms.natom() > 0, "No polarizable atoms specified"
        self.point_charges = point_charges
        
        self.polarizabilities = polarizabilities

        # number of polarizable sites
        self.npol = polarizable_atoms.natom()
        # number of basis functions
        self.nbf = self.basis.nbf()

        # cutoff function C(r) = (1- exp(-alpha r^2))^q
        self.cutoff_alpha = 4.0
        self.cutoff_power = 2

        # integrals
        self.mints = core.MintsHelper(self.basis)
        
        # compute quantities needed for constructing two-electron, one-electron and zero-electron
        # contributions to the Hamiltonian due to the presence of polarizable atoms
        self.A = self._effective_polarizability()
        self.F_elec = self._polarization_integrals()
        self.F_nucl = self._nuclear_fields()
        
        # for resolution of identity we need the inverse of the overlap matrix
        mints_ri = core.MintsHelper(self.ribasis)
        S = mints_ri.ao_overlap()
        self.Sinv = la.inv(S)
        
    def _effective_polarizability(self):
        """
        build the effective polarizability matrix

               -1
          A = B

        which is the inverse of

                   -1
          B = alpha   +  T

        It is not a polarizability in the strict sense of the word.
        """
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np
        
        B = np.zeros((3*self.npol,3*self.npol))
        for i in range(0, self.npol):
            # row indices of T(i,j) block in B matrix
            rows = np.arange(3*i,3*(i+1))
            for j in range(i, self.npol):
                # column indices of T(i,j) block in B matrix
                cols = np.arange(3*j,3*(j+1))

                if (i == j):
                    # atomic dipole polarizability of site i
                    alpha = self.polarizabilities[self.polarizable_atoms.symbol(i)]
                    
                    Tij = 1.0/alpha * np.eye(3)
                else:
                    # polarizability tensor for interaction between two dipoles
                    # at sites i and j
                    Tij = _T(R[i,:]-R[j,:])

                B[np.ix_(rows,cols)] = Tij
                # B-matrix is symmetric
                if (i != j):
                    B[np.ix_(cols,rows)] = Tij

        # invert B to get "effective polarizability"
        A = la.inv(B)

        return A
        
    def _polarization_integrals(self):
        """
        form the vector of Nbf x Nbf matrices

           (elec)
          F       = ( F  (R1), F  (R2), ..., F  (Ri), ..., F  (R_npol) )        (3 Npol) x Nbf x Nbf
                       mn       mn            mn            mn

        with the polarization integrals
                           r - R
          F  (R)     = (m|---------|n)               3 x Nbf x Nbf
           mn              |r-R|^3
        """        
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        # Fx,Fy,Fz = F[3*i,3*(i+1),:,:]
        F = np.zeros((3*self.npol, self.nbf, self.nbf))
        # enumerate polarizable sites
        for i in range(0, self.npol):
            # x-component of vector F_mn(R_i)
            F[3*i+0,:,:] = self.mints.ao_polarization(R[i,:],
                                                      3, 1,0,0, # k=3, mx=1, my=0, mz=0
                                                      self.cutoff_alpha, self.cutoff_power)
            # y-component of vector F_mn(R_i)
            F[3*i+1,:,:] = self.mints.ao_polarization(R[i,:],
                                                      3, 0,1,0, # k=3, mx=0, my=1, mz=0
                                                      self.cutoff_alpha, self.cutoff_power)
            # z-component of vector F_mn(R_i)
            F[3*i+2,:,:] = self.mints.ao_polarization(R[i,:], 
                                                      3, 0,0,1, # k=3, mx=0, my=0, mz=1
                                                      self.cutoff_alpha, self.cutoff_power)
        
        return F

    def _nuclear_fields(self):
        """
        form the vector 

            (nuclear)
           F          = ( F(R1), F(R2), ..., F(R_npol) )        dimension: (3 Nol)

        with
                           R - R_n 
           F(R) = sum Q  -----------
                   n   n |R - R_n|^3

        where Q_n are all point charges
        """
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        # Fx,Fy,Fz = F[3*i,3*(i+1),:,:]
        F = np.zeros(3*self.npol)

        # enumerate polarizable sites
        for i in range(0, self.npol):
            Fi = np.zeros(3)
            
            # add fields from nuclei
            Rnuc = self.molecule.geometry().np
            for n in range(0, self.molecule.natom()):
                Qn = self.molecule.Z(n)
                Rn = Rnuc[n,:]
                Fi += Qn * (R[i,:] - Rn)/pow(la.norm(R[i,:] - Rn), 3)


            # add fields from other point charges
            if (self.point_charges.natom() > 0):
                Rchrg = self.point_charges.geometry().np
                for n in range(0, self.point_charges.natom()):
                    Qn = self.point_charges.charge(n)
                    Rn = Rchrg[n,:]
                    Fi += Qn * (R[i,:] - Rn)/pow(la.norm(R[i,:] - Rn), 3)


            F[3*i:3*(i+1)] = Fi

        return F
    
    def coulomb_J(self, C_left, C_right):
        """
        build Coulomb operator

                          (2)      
          J   = sum  (pq|h   |rs) D
           pq   r,s                rs

        The one-particle density matrix is constructed from the molecular
        orbital coefficients as

                       (L) (R)
          D   =  sum  C   C              (k runs over occupied MOs)
           rs     k    rk  sk

        Parameters
        ----------
        C_left  :    np.ndarray    (Nbf x Nocc)
          left molecular orbital coefficients
        C_right :    np.ndarray    (Nbf x Nocc)
          right molecular orbital coefficients

        Returns
        -------
        J     :    np.ndarray    (Nbf x Nbf)
          Coulomb operator in AO basis
        """
        D = np.einsum('rk,sk->rs', C_left, C_right)
        FD = np.einsum('icd,cd->i', self.F_elec, D)
        J = -np.einsum('iab,ij,j->ab', self.F_elec, self.A, FD)
        return J

    def exchange_K(self, C_left, C_right):
        """
        build exchange operator

                          (2)      
          K   = sum  (pr|h   |qs)  D
           pq   r,s                 rs

        The one-particle density matrix is constructed from the molecular
        orbital coefficients as

                       (L) (R)
          D   =  sum  C   C              (k runs over occupied MOs)
           rs     k    rk  sk


        Parameters
        ----------
        C_left  :    np.ndarray    (Nbf x Nocc)
          left molecular orbital coefficients
        C_right :    np.ndarray    (Nbf x Nocc)
          right molecular orbital coefficients

        Returns
        -------
        K     :    np.ndarray    (Nbf x Nbf)
          exchange operator in AO basis
        """
        FC_left = np.einsum('iag,gk->iak', self.F_elec, C_left)
        FC_right = np.einsum('jbd,dk->jbk', self.F_elec, C_right)
        K = -np.einsum('iak,ij,jbk->ab', FC_left, self.A, FC_right)
        return K
        
    def two_electron_part(self):
        """
        compute correction to two-electron integrals due to presence of polarizable atoms

                       (2)          (e)     (e)
          I     = (ab|h   |cd) = - F    A  F
           abcd                     ab      cd

        Returns
        -------
        I     :   np.ndarray (Nbf x Nbf x Nbf x Nbf)
          matrix elements of two-electron part of polarization Hamiltonian
        """
        I = -np.einsum('iab,ij,jcd->abcd', self.F_elec, self.A, self.F_elec)
        return I

    def one_electron_part(self):
        """
        compute correction to core Hamiltonian due to presence of polarizable atoms

                    (1)       (n)     (e)                 -1      (e)     (e)
          V   = (a|h   |b) = F    A  F     - 1/2 sum    (S  )    F    A  F
           ab                         ab            cd       cd   ac      db

        Returns
        -------
        V    :    np.ndarray (Nbf x Nbf)
          matrix elements of one-electron part of polarization Hamiltonian
        """
        V = np.einsum('i,ij,jab->ab', self.F_nucl, self.A, self.F_elec)
        # S^(-1) F^(e)
        SinvF = np.einsum('cd,idb->icb', self.Sinv, self.F_elec)
        V -= 0.5 * np.einsum('iag,ij,jgb->ab', self.F_elec, self.A, SinvF)

        return V
        
    def zero_electron_part(self):
        """
        compute correction to zero-electron Hamiltonian

                    (n)     (n)
          E = -1/2 F    A  F

        Returns
        -------
        E   :    float
          constant energy offset (depends only on nuclear coordinates)
        """
        E = -0.5 * np.einsum('i,ij,j', self.F_nucl, self.A, self.F_nucl)
        return E


################# TESTING ##########################

def rhf_qmmm2epol(molecule, polarizable_atoms, point_charges, basis_name,
                  polarizabilities=_theoretical_polarizabilities):
    """
    perform restricted Hartree-Fock calculation for a closed-shell molecule
    with the QM/MM-2e-pol Hamiltonian. Polarizable atoms are incorporated
    in an effective manner by changing the two- and one-electron matrix elements
    of the Hamiltonian.
    
    Parameters
    ----------
    molecule          : psi4.core.Molecule
      atoms in QM part are treated explicitely with quantum mechanics
    polarizable_atoms : psi4.core.Molecule
      polarizable atoms in the MM part can be polarized by the QM part and generate
      a polarization potential, which in turn affects the electrons in the QM region.
    point_charges     : psi4.core.Molecule
      additional point charges if present
    basis_name        : str
      name of basis set, e.g. 'cc-pVDZ'
    polarizabilities   :  dict
      dictionary with atomic polarizabilities for each atom type

    Returns
    -------
    SCF_E             : float
      total SCF energy
    """
    # RHF adapted from https://github.com/humeniuka/psi4numpy/tree/master/Self-Consistent-Field
    import time
    
    time_start = time.time()

    # QM atoms
    molecule.print_out()
    # MM atoms
    polarizable_atoms.print_out()
    point_charges.print_out()
    
    # basis for QM atoms
    wfn = psi4.core.Wavefunction.build(molecule, basis_name)
    assert wfn.nalpha() == wfn.nbeta(), "only works for closed-shell molecules"
    basis = wfn.basisset()
    ribasis = basis

    # Polarization Hamiltonian
    if polarizable_atoms.natom() > 0:
        polham = PolarizationHamiltonian(molecule, basis, ribasis,
                                         polarizable_atoms, point_charges,
                                         polarizabilities=polarizabilities)
    else:
        polham = None

    mints = psi4.core.MintsHelper(basis)
    
    # Build H_core
    V = mints.ao_potential().np
    T = mints.ao_kinetic().np
    H = V + T
    # nuclear energy
    Enuc = molecule.nuclear_repulsion_energy()

    if not polham is None:
        # add one-electron part of polarization potential to core Hamiltonian
        Vpol = polham.one_electron_part()
        H += Vpol
        # nuclear part of polarization Hamiltonian
        Enuc += polham.zero_electron_part()
        
    # overlap matrix in AO basis
    S = mints.ao_overlap().np
    # Loewdin orthogonalization
    # X^{-1} = S^{1/2}
    Xinv = sla.sqrtm(S)
    # X = S^{-1/2}
    X = la.inv(Xinv)
    
    # Initialize the JK object
    jk = psi4.core.JK.build(basis, jk_type="DF")
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()

    # number of occupied orbitals
    ndocc = wfn.nalpha()

    # Calculate initial guess for MOs by diaginalizing core Hamiltonian
    e, C = sla.eigh(H, S)
    # select coefficients of doubly occupied orbitals
    Cocc = C[:, :ndocc]
    
    # initial density matrix
    D = 2 * np.einsum('pi,qi->pq', Cocc, Cocc)
    
    # Set defaults
    maxiter = 100
    E_conv = 1.0E-10
    diis_conv = 1.0E-6

    Eold = 0.0

    # Pulay's direct inversion in iterative subspace
    diis = psi4.p4util.solvers.DIIS()
    
    for SCF_ITER in range(1, maxiter + 1):

        # Compute JK
        jk.C_left_add(core.Matrix.from_array(Cocc))
        jk.compute()
        jk.C_clear()

        J = jk.J()[0].np
        K = jk.K()[0].np

        # add (\Delta J) and (\Delta K) from polarization Hamiltonian
        if not polham is None:
            J += polham.coulomb_J(Cocc, Cocc)
            K += polham.exchange_K(Cocc, Cocc)

        # Build Fock matrix in AO basis
        F = H + 2*J - K

        # check that MO coefficients are orthogonal with respect to overlap matrix
        #  C^T . S . C = Id
        assert la.norm(np.dot(Cocc.T, np.dot(S, Cocc))  - np.eye(ndocc)) < 1.0e-5
        
        # see
        #   Pulay, Peter. "Improved SCF convergence acceleration."
        #   Journal of Computational Chemistry 3.4 (1982): 556-560.
        #
        # error vector
        error = F.dot(D).dot(S) - S.dot(D).dot(F)
        error = X.dot(error).dot(X)
        
        diis.add(core.Matrix.from_array(F), core.Matrix.from_array(error))

        dRMS = la.norm(error)
        
        # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
        SCF_E = 0.5 * np.einsum('pq,pq->', F + H, D) + Enuc

        dE = abs(SCF_E - Eold)
        print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E' % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
        if (abs(dE < E_conv) and (dRMS < diis_conv)):
            break

        Eold = SCF_E

        if (SCF_ITER > 1):
            F = diis.extrapolate().np

        # Diagonalize Fock matrix by solving the generalized eigenvalue problem
        #   F C = e S C
        e, C = sla.eigh(F, S)
        # MO coefficients of doubly occupied orbitals
        Cocc = C[:,:ndocc]
        # density matrix in AO basis
        D = 2 * np.einsum('ai,bi->ab', Cocc, Cocc)
        #print("number of lectrons = ", np.sum(D*S))
        
        # Is the density matrix idempotent ?
        err = 0.5*D - (0.5*D).dot(S).dot(0.5*D)
        assert la.norm(err) < 1.0e-10, "Density matrix not idempotent, |D - D.S.D| > 0"

        
        if SCF_ITER == maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - time_start))

    return SCF_E
    
    
if __name__ == "__main__":
    import time
    import psi4
    psi4.set_memory('500 MB')

    #
    # Li+ (QM) ----- He (CPP)
    #
    
    molecule = psi4.geometry("""
    +1 1
    Li  0.000000  0.000000  0.000000

    units angstrom
    no_com
    no_reorient
    symmetry c1
    """)

    polarizable_atoms = psi4.geometry("""
    He  4.000000  0.000000  0.000000

    units angstrom
    no_com
    no_reorient
    symmetry c1
    """)

    point_charges = psi4.geometry("""
    """)
    
    # run closed-shell SCF calculation
    SCF_E = rhf_qmmm2epol(molecule, polarizable_atoms, point_charges, 'cc-pVDZ')
    print('Final QMMM-2e-pol SCF energy : %.8f hartree' % SCF_E)
    
    # compare with Xiao's code
    SCF_E_reference = -7.236167358544909
    print('expected                     : %.8f hartree' % SCF_E_reference)
    error_eV = (SCF_E - SCF_E_reference) * 27.211
    print("error = %.8f meV" % (error_eV * 1000))
    assert abs(SCF_E - SCF_E_reference) < 1.0e-6

