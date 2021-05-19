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

References
----------
[CPP]    Schwerdtfeger, P., and H. Silberbach. 
         "Multicenter integrals over long-range operators using Cartesian Gaussian functions." 
         Phys. Rev. A 37.8 (1988): 2834.
[Meyer]  Mueller, Wolfgang, Joachim Flesch, and Wilfried Meyer. 
        "Treatment of intershell correlation effects in abinitio calculations by use of core polarization potentials. 
          Method and application to alkali and alkaline earth atoms." 
         J. Chem. Phys. 80.7 (1984): 3297-3310.
[Thole]  B.T. Thole, 
        "Molecular polarizabilities calculated with a modified dipole interaction."
         Chem. Phys. 59 (1981), 341-350.

"""
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import itertools

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
                 polarizabilities=_theoretical_polarizabilities,
                 same_site_integrals='exact',
                 dipole_damping='Thole',
                 monopole_damping='Thole'):
        """
        two-electron, one-electron and zero-electron contributions to electronic
        Hamiltonian due to the presence of polarizable sites

        Parameters
        ----------
        molecule            :  psi4.core.Molecule
          QM part
        basis               :  psi4.core.BasisSet
          basis set for QM part
        ribasis             :  psi4.core.BasisSet
          auxiliary basis set for QM part (used for resolution of identity)
        polarizable_atoms   :  psi4.core.Molecule
          polarizable atoms in MM part
        point_charges       :  psi4.core.Molecule
          MM atoms which carry point charges
        polarizabilities    :  dict
          dictionary with atomic polarizabilities for each atom type
        same_site_integrals :  str
          'exact' - use analytically exact polarization integrals whenever possible
          'R.I.'  - treat all integrals on the same footing by using the resolution of identity
          This only affects the 1e part of the Hamiltonian.
        dipole_damping      :  str
          The classical dipole polarizability may diverge for certain geometries, unless the dipole-dipole
          interaction is damped at close distances. The following damping functions are available:
            * 'Thole'  - the dipole field tensor is modified according to eqns. A.1 and A.2 in [Thole].
            * 'CPP'    - the same cutoff function as for the CPP integrals is used (see eqn. 4 in [CPP]) (not recommended)
            *  None    - no damping at all  (not recommended)
        monompole_damping   :  str
          The field of a point charge at the position of a polarizable atom becomes infinite as
          the two approach. This can be avoided if the point charge (QM nuclei and MM partial charges)
          are replaced by smeared out charge distributions so that the field approaches a finite value 
          as the point charge and the polarizable atom fuse.
            * 'Thole'  - replace the field of a point charge by eqn. A4 in [Thole]
            * None     - use field of a point charge, E(i) = Q(i) * r(i)/r^3   (not recommended)
        """
        self.molecule = molecule
        self.basis = basis
        # auxiliary basis set for resolution of identity
        self.ribasis = ribasis
        
        self.polarizable_atoms = polarizable_atoms
        assert self.polarizable_atoms.natom() > 0, "No polarizable atoms specified"
        self.point_charges = point_charges
        
        self.polarizabilities = polarizabilities
        print(f"same-site polarization integrals are treated by method : '{same_site_integrals}'")
        self.same_site_integrals = same_site_integrals
        self.dipole_damping = dipole_damping
        print(f"damping of dipole-dipole interaction : {dipole_damping}")
        self.monopole_damping = monopole_damping
        print(f"damping of monopole field            : {monopole_damping}")
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
        self.alpha = self._atomic_polarizabilities()
        self.A = self._effective_polarizability()
        self.F_elec = self._polarization_integrals_F()
        self.I_ee = self._polarization_integrals_I()
        self.F_nucl = self._nuclear_fields()

        # If the polarizable atoms were replaced by a single polarizable site, what
        # would be its polarizability tensor?
        self._molecular_polarizability()
        
        # for resolution of identity we need the inverse of the overlap matrix
        mints_ri = core.MintsHelper(self.ribasis)
        S = mints_ri.ao_overlap()
        self.Sinv = la.inv(S)

    def _atomic_polarizabilities(self):
        """
        assemble vector of atomic polarizabilities

          alpha = (alpha , ..., alpha , ..., alpha    )
                        1            i            npol

        Returns
        -------
        alpha   :  np.ndarray (Npol,)
          atomic polarizabilities in bohr^3
        """
        alpha = np.zeros(self.npol)
        for i in range(0, self.npol):
            # atomic dipole polarizability of site i
            alpha[i] = self.polarizabilities[self.polarizable_atoms.symbol(i)]
        return alpha
            
    def _effective_polarizability(self, dipole_damping=None):
        """
        build the effective polarizability matrix

               -1
          A = B

        which is the inverse of

                   -1
          B = alpha   +  T

        It is not a polarizability in the strict sense of the word.

        Parameters
        ----------
        dipole_damping  :  str
          name of damping function for modifying the dipole field tensor T(r) at short distances.
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
                    Tij = 1.0/self.alpha[i] * np.eye(3)
                else:
                    rvec = R[i,:]-R[j,:]
                    # polarizability tensor for interaction between two dipoles
                    # at sites i and j
                    Tij = _T(rvec)

                    # The dipole polarizability may diverge leading to the polarization catastrophe
                    # unless the dipole-dipole interaction between point dipoles is damped.
                    
                    # The dipole field tensor can be derived as the second derivative tensor of
                    # the electrostatic potential generated by some spherically symmetric charge
                    # distribution. Thole found that a conical charge distribution
                    # (rho4 in table 1 of [Thole]) gives the best fit for molecular polarizabilities.
                    if dipole_damping == 'Thole':
                        r = la.norm(rvec)
                        # see text above eqn. A.1 in [Thole]
                        s = 1.662 * pow(self.alpha[i]*self.alpha[j], 1.0/6.0)
                        if (r < s):
                            v = r/s
                            # damped dipole-dipole interaction according to eqn. A.2 in [Thole]
                            # For v -> 0, T(v) -> 0
                            Tij = 4*pow(v,3)*(1.0-v) * np.eye(3)/pow(r,3) + pow(v,4) * Tij
                    elif dipole_damping == 'CPP':
                        r = la.norm(rvec)
                        # cutoff function from CPP integrals
                        cutoff = pow(1.0 - np.exp(-self.cutoff_alpha * r**2), self.cutoff_power)
                        Tij *= cutoff
                        
                B[np.ix_(rows,cols)] = Tij
                # B-matrix is symmetric
                if (i != j):
                    B[np.ix_(cols,rows)] = Tij

        # invert B to get "effective polarizability"
        A = la.inv(B)

        return A
        
    def _molecular_polarizability(self):
        """
        The molecular polarizability tells us the total dipole moment induced
        in a molecule by an external field. 

          mu    =  alpha    E
            tot         mol

        see eqn. (6) in [Thole]
        """
        # The dipole moment mu(i)  induced on atom i by the fields E(j) at all atom j is
        #
        #    mu(i) = sum  A(i,j) E(j)
        #               ij
        # Assuming that the field is the same at all atom, E(i)=E, the total dipole moment is
        #
        #    mu(tot) = sum  mu(i) = sum   A(i,j)  E
        #                 i            ij
        #
        # Therefore the molecular dipole polarizability is given by
        #
        # alpha_mol = sum_ij A(i,j).
        #
        alpha_mol = np.zeros((3,3))
        for i in range(0, self.npol):
            for j in range(0, self.npol):
                alpha_mol += self.A[3*i:3*(i+1),3*j:3*(j+1)]

        # abbreviation
        a = alpha_mol
        print(f""" 

   Total dipole polarizability alpha_mol of MM part (in bohr^3)

               X        Y        Z
        X  {a[0,0]:+8.4f} {a[0,1]:+8.4f} {a[0,2]:+8.4f} 
        Y  {a[1,0]:+8.4f} {a[1,1]:+8.4f} {a[1,2]:+8.4f} 
        Y  {a[2,0]:+8.4f} {a[2,1]:+8.4f} {a[2,2]:+8.4f} 
        """)

        return alpha_mol
        
    def _polarization_integrals_F(self):
        """
        form the vector of Nbf x Nbf matrices

           (elec)
          F       = ( F  (R1), F  (R2), ..., F  (Ri), ..., F  (R_npol) )        (3 Npol) x Nbf x Nbf
                       mn       mn            mn            mn

        with the polarization integrals
                           r - R
          F  (R)     = (m|--------- Cutoff(|r-R|) |n)               3 x Nbf x Nbf
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

    def _polarization_integrals_I(self):
        """
        form the vector of Nbf x Nbf matrices

           (ee)
          I    = ( I  (R1), ..., I  (Ri), ..., I  (R_npol) )          Npol x Nbf x Nbf
                    mn            mn            mn

        with the polarization integrals 
                          1                 2
          I  (R) = (m| ------- Cutoff(|r-R|)  |n)         Nbf x Nbf
           mn          |r-R|^4
        """
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np
        # I^{(ee)}(R(i)) = I[i,:,:]
        I = np.zeros((self.npol,self.nbf, self.nbf))
        for i in range(0, self.npol):
            I[i,:,:] = self.mints.ao_polarization(R[i,:],
                                                  4, 0,0,0, # k=4, mx=my=mz=0,
                                                  self.cutoff_alpha, 2*self.cutoff_power)
        return I
    
    def _nuclear_fields(self):
        """
        form the vector 

            (nuclear)
           F          = ( F(R1), F(R2), ..., F(R_npol) )        dimension: (3 Npol)

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
                rvec = R[i,:] - Rn
                r = la.norm(rvec)

                if self.monopole_damping == 'Thole':
                    # see text above eqn. A.1 in [Thole]
                    # Since we do not have a polarizability for every nucleus, we use
                    # the same value as for the polarizable atom.
                    s = 1.662 * pow(self.alpha[i]*self.alpha[i], 1.0/6.0)
                    if r == 0.0:
                        # The limit of the damped field is lim_r->0 E(r) = 0.0
                        dFn = 0*rvec
                    else:
                        dFn = Qn * rvec/pow(r,3)
                        if (r < s):
                            v = r/s
                            # see eqn. A.4 in [Thole]
                            # The electric field at the nucleus is damped.
                            f = (4*pow(v,3) - 3*pow(v,4))
                            dFn *= f
                else:
                    # undamped field of a monopole
                    dFn = Qn * rvec/pow(r,3)

                Fi += dFn

            # add fields from other point charges
            if (self.point_charges.natom() > 0):
                Rchrg = self.point_charges.geometry().np
                for n in range(0, self.point_charges.natom()):
                    Qn = self.point_charges.charge(n)
                    Rn = Rchrg[n,:]
                    rvec = R[i,:] - Rn
                    r = la.norm(rvec)

                    if self.monopole_damping == 'Thole':
                        # see text above eqn. A.1 in [Thole]
                        s = 1.662 * pow(self.alpha[i]*self.alpha[i], 1.0/6.0)
                        if r == 0.0:
                            # The limit of the damped field is lim_r->0 E(r) = 0.0
                            dFn = 0*rvec
                        else:
                            dFn = Qn * rvec/pow(r,3)
                            if (r < s):
                                v = r/s
                                # see eqn. A.4 in [Thole]
                                # The electric field at the nucleus is damped.
                                dFn *= (4*pow(v,3) - 3*pow(v,4))
                    else:
                        # undamped field of a monopole
                        dFn = Qn * rvec/pow(r,3)

                    Fi += dFn

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

                    (1)         (n)     (e)                 -1      (e)     (e)
          V   = (a|h   |b) = - F    A  F     - 1/2 sum    (S  )    F    A  F
           ab                           ab            cd       cd   ac      db

                             -1     (e)            (e)                        (ee)
            +  [ 1/2 sum   (S  )   F    diag  (A) F     -  1/2 sum  alpha(i) I    (R ) ]
                        cd      cd  ac      3x3    db             i           ab    i

        The term in braces [ ... ] is only added if exact integrals are requested for
        same-site polarization integrals.

        Returns
        -------
        V    :    np.ndarray (Nbf x Nbf)
          matrix elements of one-electron part of polarization Hamiltonian
        """
        Vne = -np.einsum('i,ij,jab->ab', self.F_nucl, self.A, self.F_elec)
        # S^(-1) F^(e)
        SinvF = np.einsum('cd,idb->icb', self.Sinv, self.F_elec)
        Vee = -0.5 * np.einsum('iac,ij,jcb->ab', self.F_elec, self.A, SinvF)

        if self.same_site_integrals == "exact":
            # These integrals are not really exact, because it's assumed that the diagonal 3x3 blocks
            # are diagonal themselves: 
            #
            #   A  = diag{alpha(i),alpha(i),alpha(i)}
            #    ii
            #
            # However, this is not the case because A = (alpha^-1 + T)^{-1} and unless T=0 the
            # diagonal can mix with the other blocks. The integrals are exact only if there is
            # a single polarizable site.
            
            # only 3x3 blocks on the diagonal of A where i=j
            blocks3x3ii = [self.A[i*3:(i+1)*3,:][:,i*3:(i+1)*3] for i in range(0, self.npol)]
            Aii = sla.block_diag(*blocks3x3ii)
            # only i=j terms with resolution of identity
            Vee_ii_ri    = -0.5 * np.einsum('iac,ij,jcb->ab', self.F_elec, Aii, SinvF)
            # only i=j terms computed using exact integrals assuming Aii is equal to the
            # isotropic atomic polarizability tensor.
            Vee_ii_exact = -0.5 * np.einsum('i,iab->ab', self.alpha, self.I_ee)
            # subtract out i=j terms that were treated with resolution of identity
            # and replace them by the exact integrals
            Vee += -Vee_ii_ri + Vee_ii_exact
            
        V = Vne + Vee

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

    def two_electron_part_MO(self, C):
        """
        transform two-electron part of polarization Hamiltonian to the MO basis.

        This is not as expensive as transforming the electron-repulsion integrals,
        since the polarization integrals factorize. So we only have to transform

            (e)         (e)
           F     = sum  F    C   C
            ij    a,b   ab   ai  aj

        Parameters
        ----------
        C   :  np.ndarray (Nbf x Nmo)
          molecular orbital coefficients

        Returns
        -------
        I   :  np.ndarray (Nmo x Nmo x Nmo x Nmo)
          matrix elements in the MO basis
        """
        # transform F from AO to MO basis
        F_elec = np.einsum('xab,ai,bj->xij', self.F_elec, C, C)
        # integrals are built in the same way as in the AO basis
        I = -np.einsum('iab,ij,jcd->abcd', F_elec, self.A, F_elec)
        return I


class SCFNotConverged(Exception):
    pass
    
class RHF_QMMM2ePol(object):
    def __init__(self, molecule, polarizable_atoms, point_charges,
                 basis='cc-pVDZ',
                 continue_anyway=False, **kwds):
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
        basis             : str or psi4.core.BasisSet
          name of basis set, e.g. 'cc-pVDZ', or basis set object
        continue_anyway   :  bool
          If True, no error is raised if the SCF cycle does not converge. 
          For an FCI calculation any orthonormal basis of orbitals will do.

        The other keywords **kwds are passed to the `PolarizationHamiltonian` constructor 
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
        wfn = psi4.core.Wavefunction.build(molecule, basis)
        assert wfn.nalpha() == wfn.nbeta(), "only works for closed-shell molecules"
        basis = wfn.basisset()
        ribasis = basis

        # Polarization Hamiltonian
        if polarizable_atoms.natom() > 0:
            polham = PolarizationHamiltonian(molecule, basis, ribasis,
                                             polarizable_atoms, point_charges,
                                             **kwds)
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
            print('SCF Iteration %3d: Energy = %20.16f   dE = %10.5E   dRMS = %10.5E' % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
            if (abs(dE < E_conv) and (dRMS < diis_conv)):
                print("SCF CONVERGED")
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
                if continue_anyway == False:
                    psi4.core.clean()
                    raise SCFNotConverged("Maximum number of SCF cycles exceeded.")

        print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - time_start))

        # store QM geometry
        self.molecule = molecule
        # number of atomic and molecular orbitals
        self.nao, self.nmo = C.shape
        # number of electrons
        self.nelec = 2*ndocc
        # integral 
        self.mints = mints
        self.jk = jk
        # MO coefficients
        self.C = C
        # Hcore
        self.H = H
        # overlap matrix
        self.S = S
        # total SCF energy
        self.SCF_E = SCF_E
        # nuclear energy
        self.Enuc = Enuc
        # polarization Hamiltonian
        self.polham = polham    
        
    @property
    def energy(self):
        """
        total SCF energy (in Hartree)
        """
        return self.SCF_E

    def _frozen_core_approximation(self, Ccore):
        """
        compute frozen core energy and core-valence Hamiltonian

        In the frozen core approximation the core electrons (1s) are removed for subsequent
        correlated caculation with only the valence electrons. The core-core interaction is
        added to the nuclear energy (Efzc) and the interaction between the valence electrons
        and the closed-shell core is incorporated as an additional term in the one-electron
        part of the Hamiltonian.

        Parameters
        ----------
        Ccore      :    np.ndarray (Nfc x Ncore)
          molecular orbital coefficients of core orbitals (double occupied)

        Returns
        -------
        Efzc       :    float
          Hartree-Fock energy of closed-shell cores
        h1_fzc     :    np.ndarray (Nbf x Nbf)
          effective Hamiltonian for core-valence interaction in AO basis
        """
        # density matrix of core orbitals
        Dcore = 2 * np.einsum('ai,bi->ab', Ccore, Ccore)

        # Compute JK for core Fock matrix
        jk = self.jk
        jk.C_left_add(core.Matrix.from_array(Ccore))
        jk.compute()
        jk.C_clear()

        J = jk.J()[0].np
        K = jk.K()[0].np
        
        # add (\Delta J) and (\Delta K) from polarization Hamiltonian
        if not self.polham is None:
            J += self.polham.coulomb_J(Ccore, Ccore)
            K += self.polham.exchange_K(Ccore, Ccore)

        # Fock matrix constructed only from the density of the core electrons
        # F[Dcore]
        Fcore = self.H + 2*J - K
        # core-valence Hamiltonian in AO basis
        h1_fzc = Fcore - self.H

        # frozen core energy, core-core interaction
        Efzc = 0.5 * np.einsum('pq,pq', Fcore + self.H, Dcore)

        return Efzc, h1_fzc

        
    def hamiltonian_MO(self, frozen_core=True):
        """
        transform one- and two-electron part of Hamiltonian from the 
        AO to the MO basis, which is orthonormal.

        Parameters
        ----------
        frozen_core   :  bool
          If True, 1s core orbitals are replaced by an effective Hamiltonian

        Returns
        -------
        h0   :  float
          core energy (nuclear repulsion + HF energy of frozen core electrons)
        h1   :  np.ndarray (Nmo x Nmo)
          1e MO matrix elements (i|h^(1)|j)
        h2   :  np.ndarray (Nmo x Nmo x Nmo x Nmo)
          2e MO matrix elements (ij|h^(1)|kl)
        """
        # == AO to MO transformation ==
        # transform one-electron part (Hcore), which already includes the contribution
        # from the polarization integrals.
        h1 = np.einsum('ab,ai,bj->ij', self.H, self.C, self.C)
        # transform electron repulsion integrals, mints expects coefficients as core.Matrix 
        C = core.Matrix.from_array(self.C)
        h2 = self.mints.mo_eri(C, C, C, C).np
        if not (self.polham is None):
            # add two-electron part from polarization
            h2 += self.polham.two_electron_part_MO(self.C)

        if frozen_core:
            # count number of core orbitals
            ncore = 0
            for i in range(0, self.molecule.natom()):
                if self.molecule.Z(i) > 2:
                    # elements in the second row have double occupied 1s core orbitals
                    ncore += 1
            self.ncore = ncore

            # MO coefficients of core orbitals
            Ccore = self.C[:,:ncore]

            # == Frozen Core Approximation ==
            Efzc, h1_fzc_ao = self._frozen_core_approximation(Ccore)
            # transform Hamiltonian for interaction between frozen core
            # and valence electrons to MO basis
            h1_fzc = np.einsum('ab,ai,bj->ij', h1_fzc_ao, self.C, self.C)
            # add interaction between valence electrons and frozen core to one-electron part
            h1 += h1_fzc
            
            print(f"number of core orbitals  : {ncore}")
            print(f"frozen core energy : {Efzc} Hartree")
            nc = ncore
            
            h0 = self.Enuc + Efzc
            h1 = h1[nc:,nc:]
            h2 = h2[nc:,nc:,nc:,nc:]

        else:
            # no core electrons, core energy is equal to repulsion between nuclei
            self.ncore = 0
            h0 = self.Enuc
            
        return h0, h1, h2
                
    def fcidump(self, filename="/tmp/hamiltonian.FCIDUMP", frozen_core=True,
                MS2=0, NROOT=1, ISYM=1, **kwds):
        """
        save matrix elements of Hamiltonian in MO basis in the format understood
        by Knowles' and Handy's Full CI program. The structure of the input data
        is described in
        
          [FCI] Knowles, Peter J., and Nicholas C. Handy. 
                "A new determinant-based full configuration interaction method." 
                Chem. Phys. Lett. 111(4-5) (1984): 315-321.

        Parameters
        ----------
        filename      :  str
          path to FCIDUMP output file
        frozen_core   :  bool
          If True, 1s core orbitals are replaced by an effective Hamiltonian

        Additional variables for the header of the FCI programm can be 
        specified as keywords. See section 3.3.1 in [FCI] for a list of 
        all variables.

        Keywords
        --------
        MS2   :  int
          total spin
        NROOT :  int
          number of states in the given spin and symmetry space
        ISYM  :  int
          spatial symmetry of wavefunction
        ...
        """
        import fortranformat
        formatter=fortranformat.FortranRecordWriter('(E24.16,I4,I4,I4,I4/)')
        
        h0, h1, h2 = self.hamiltonian_MO(frozen_core=frozen_core)
        with open(filename, "w") as f:
            # header
            f.write(f"$FCI NORB={self.nmo-self.ncore},NELEC={self.nelec-2*self.ncore},\n")
            f.write(f" MS2={MS2},ISYM={ISYM},NROOT={NROOT},\n")
            for key,value in kwds.items():
                # additional keywords
                f.write(f" {key}={value},\n")
            f.write("$END\n")
            # number of valence orbitals
            nmo = self.nmo-self.ncore
            # enumerate two-electron integrals
            for i in range(0,nmo):
                for j in range(0, nmo):
                    for k in range(0, nmo):
                        for l in range(0, nmo):
                            f.write(formatter.write([ h2[i,j,k,l], i+1,j+1,k+1,l+1 ]))
            # enumerate one-electron integrals
            for i in range(0, nmo):
                for j in range(0, nmo):
                    f.write(formatter.write([ h1[i,j], i+1,j+1, 0, 0 ]))
            # core energy
            f.write(formatter.write([ h0, 0, 0, 0, 0 ]))
                
            
if __name__ == "__main__":
    import time
    import psi4
    psi4.set_memory('500 MB')

    #
    # Li+ (QM) ----- He (MM)
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
    rhf = RHF_QMMM2ePol(molecule, polarizable_atoms, point_charges,
                        basis='cc-pVDZ',
                        same_site_integrals='exact') # 'R.I.'
    SCF_E = rhf.energy
    print('Final QMMM-2e-pol SCF energy : %.8f Hartree' % SCF_E)

    # compare with Xiao's code
    SCF_E_reference = -7.236167358544909
    print('expected                     : %.8f Hartree' % SCF_E_reference)
    error_eV = (SCF_E - SCF_E_reference) * 27.211
    print("error = %.8f meV" % (error_eV * 1000))
    assert abs(SCF_E - SCF_E_reference) < 1.0e-6

    # export matrix elements in MO basis for Peter Knowles' full CI program
    #rhf.fcidump(filename="/tmp/Li+_QM__He_MM.fcidump", NROOT=2)
