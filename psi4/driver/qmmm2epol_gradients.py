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
Analytical gradients for QM/MM-2e-Pol

There are three types of derivatives depending on which type of atomic centers are moved:
Derivatives with respect to

 * QM atoms
 * point charges 
 * polarizable sites

The three sets are not disjoint. The QM nuclei belong to the point charges and are centers
of basis functions at the same time. Similarly, some of the point charges may also have polarizabilities. 
However, the gradients are computed separately for each group of centers. 
It is only at the end that the different gradient contributions are added for the same centers.

The type of gradient computed by a member function is indicated by its name

 * QM atoms           -->    some_quantity_DERIV_QM(k,  c)
 * point charges      -->    some_quantity_DERIV_CHG(k, c)
 * polarizable sites  -->    some_quantity_DERIV_POL(k, c)

`k` is the index of the center in the respective group and `c` enumerates the Cartesian coordinate (x,y or z).
Derivatives are computed for each atom coordinate separately, so that the dimensions of the tensors,
which are differentiated, do not change. 

To avoid having to keep large arrays in memory, the gradients of the effective polarizability A, diag(A),
the polarization integrals F^(elec), I^(elec) and the overlap matrix S are directly contracted with
the partial derivatives. For instance the gradient of a function

   f(A, F^(e), F^(n), I^(e), ...)

is computed as

   df   d A   df    d F^(e)   d f
   -- = --- . -- +  ------- . -------  +  ....
   dx   d x   dA      d x     d F^(e)

where the scalar products are implemented as linear functions operating on tensors, i.e. dA/dx(U) with U=df/dA.
For each gradient contraction a member function with the prefix _grad###() exists, e.g. `_gradA(U)` or `_gradFelec(E)`.

These linear functions return the gradients on QM atoms, polarizable sites and point charges as one long
vector of size `3*(nat+npol+nchg)`.

"""
import numpy as np
import numpy.linalg as la

import psi4
from psi4 import core
from psi4.driver.qmmm2epol import PolarizationHamiltonian, RHF_QMMM2ePol

class PolarizationHamiltonianGradients(PolarizationHamiltonian):
    """
    gradients of two-electron, one-electron and zero-electron parts of the polarization Hamiltonian
    """
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        # initialize various indices
        self._gradient_dimensions()

    def _gradient_dimensions(self):
        """
        The gradient vector has length 

           ngrad = 3*nat + 3*npol + 3*nchg 

        The three blocks contain the gradient with respect to

           *  the QM atoms,            grad[0 : 3*nat                                 ]
           *  the polarizable sites,   grad[    3*nat : 3*(nat+npol)                  ]
           *  the point charges,       grad[            3*(nat+npol):3*(nat+npol+nchg)]
        
        This functions determines the sizes and starting offsets of the different blocks.
        """
        self.nat = self.molecule.natom()
        # already set in super class
        #self.npol = self.polarizable_atoms.natom()
        self.nchg = self.point_charges.natom()

        # length of gradient vector
        self.ngrad = 3*(self.nat + self.npol + self.nchg)

        # start of QM gradient
        self.startQM = 0
        # start of gradient on polarizable sites
        self.startPOL = 3*self.nat
        # start of gradient on point charges
        self.startCHG = 3*(self.nat+self.npol)

    def split_gradient(self, grad):
        """
        split gradients into blocks for QM atoms, polarizable sites and point charges
        """
        assert len(grad) == self.ngrad
        grad_QM  = grad[self.startQM :self.startPOL]
        grad_POL = grad[self.startPOL:self.startCHG]
        grad_CHG = grad[self.startCHG:             ]
        return grad_QM, grad_POL, grad_CHG

    def _damping_function(self, r, alpha):
        # undamped field of a monopole
        f = 1.0
        df = 0.0
        if self.monopole_damping == 'Thole':
            # see text above eqn. A.1 in [Thole]
            # Since we do not have a polarizability for every nucleus, we use
            # the same value as for the polarizable atom.
            s = 1.662 * pow(alpha*alpha, 1.0/6.0)
            if r == 0.0:
                # The limit of the damped field is lim_r->0 E(r) = 0.0
                return 0.0, 0.0
            else:
                if (r < s):
                    v = r/s
                    # see eqn. A.4 in [Thole]
                    # The electric field at the nucleus is damped.
                    f = (4*pow(v,3) - 3*pow(v,4))
                    # derivative df(r)/dr
                    df = (12*pow(v,2) - 12*pow(v,3))/s

        # The nuclear fields have to be damped by the same cutoff function as the electronic fields.
        # Otherwise the fields from protons and electrons do not cancel at long range as
        # expected for a neutral molecule.
        a = self.cutoff_alpha
        q = self.cutoff_power
        # cutoff function C(r)
        C = pow(1.0 - np.exp(-a*r**2), q)
        # derivative dC(r)/dr
        dC = 2*a*q*r * pow(1.0 - np.exp(-a*r**2), q-1) * np.exp(-a*r**2)
        
        damp = f*C
        damp_deriv = df*C + f*dC

        return damp, damp_deriv

    def _field_deriv(self, rvec, c, polarizability):
        """
        derivative of 

            rvec
           ------ damp(r)
            |r|^3

        with respect to coordinate rvec[c] of rvec.
        The damping function is defined in terms of a polarizability (Thole's damping).
        """
        r = la.norm(rvec)
        dF = np.zeros(3)
        if (r == 0.0):
            # The damping function removes the divergence at r = 0.0
            return dF

        damp, damp_deriv = self._damping_function(r, polarizability)

        # loop over x,y,z
        for a in [0,1,2]:
            if (a == c):
                dF[a] += 1/pow(r,3)
            dF[a] -= 3 * rvec[a]*rvec[c]/pow(r,5)
            dF[a] *= damp
            dF[a] += rvec[a]*rvec[c]/pow(r,4) * damp_deriv

        return dF

    def _F_nucl_DERIV_QM(self, inuc, c):
        """derivative of nuclear fields at the polarizable sites with respect to the positions of the nuclei"""
        dF = np.zeros(3*self.npol)

        # cartesian coordinates of polarizable sites
        Rpol = self.polarizable_atoms.geometry().np
        # and of nuclei
        Rnuc = self.molecule.geometry().np

        # add fields from nuclei
        # loop over polarizable sites
        for i in range(0, self.polarizable_atoms.natom()):
            Q = self.molecule.Z(inuc)
            rvec = Rpol[i,:] - Rnuc[inuc,:]
            # add contribution from this nucleus
            dF[3*i:3*(i+1)] -= Q * self._field_deriv(rvec, c, self.alpha[i])

        return dF

    def _F_nucl_DERIV_CHG(self, ichrg, c):
        """derivative of nuclear fields at the polarizable sites with respect to the positions of the point charges"""
        dF = np.zeros(3*self.npol)
        # cartesian coordinates of polarizable sites
        Rpol = self.polarizable_atoms.geometry().np
        # cartesian coordinates of point charges
        Rchrg = self.point_charges.geometry().np

        # add fields from point charges
        # loop over polarizable sites
        for i in range(0, self.polarizable_atoms.natom()):
            Q = self.point_charges.Z(ichrg)
            rvec = Rpol[i,:] - Rchrg[ichrg,:]
            # add contribution from point charge ichrg
            dF[3*i:3*(i+1)] -= Q * self._field_deriv(rvec, c, self.alpha[i])

        return dF

    def _F_nucl_DERIV_POL(self, ipol, c):
        """derivative of nuclear fields at the polarizable sites with respect to the positions of the polarizable sites"""
        # cartesian coordinates of polarizable sites
        Rpol = self.polarizable_atoms.geometry().np

        dFi = np.zeros(3)

        # add fields from nuclei
        Rnuc = self.molecule.geometry().np
        # loop over nuclei
        for n in range(0, self.molecule.natom()):
            Q = self.molecule.Z(n)
            rvec = Rpol[ipol,:] - Rnuc[n,:]
            # add contribution from nucleus n
            dFi += Q * self._field_deriv(rvec, c, self.alpha[ipol])

        # add fields from point charges
        if (self.point_charges.natom() > 0):
            Rchrg = self.point_charges.geometry().np
            # loop over point charges
            for n in range(0, self.point_charges.natom()):
                Q = self.point_charges.Z(n)
                rvec = Rpol[ipol,:] - Rchrg[n,:]
                # add contribution from point charge n
                dFi += Q * self._field_deriv(rvec, c, self.alpha[ipol])

        dF = np.zeros(3*self.npol)
        dF[3*ipol:3*(ipol+1)] = dFi

        return dF

    def _T_DERIV_POL(self, k, c):
        """
        derivative of the dipole field tensor with respect to the c-th coordinate of the 
        k-th polarizable site.

        Parameters
        ----------
        k    :   int 
          index of polarizable site
        c    :   0,1 or 2
          Cartesian coordinate (x,y or z)

        Returns
        -------
        dT   :   (3*npol, 3*npol) matrix
          derivative dT/d(R_{k,c})
        """
        # cartesian coordinates of polarizable sites
        Rpol = self.polarizable_atoms.geometry().np
        
        dT = np.zeros((3*self.npol, 3*self.npol))
        for i in range(0, self.npol):
            row = 3*i
            for j in range(0, self.npol):
                col = 3*j
                if (k == i or k == j) and (i != j):
                    rvec = Rpol[i,:]-Rpol[j,:]
                    # |rvec|
                    r2 = rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2];
                    r = np.sqrt(r2);

                    # No damping
                    damp3 = 1.0
                    damp5 = 1.0
                    # If there is no damping, derivatives of damping factors vanish
                    damp3_DERIV = 0.0
                    damp5_DERIV = 0.0
                    
                    if self.dipole_damping == 'Thole':
                        # see text above eqn. A.1 in [Thole]
                        s = 1.662 * pow(self.alpha[i]*self.alpha[j], 1.0/6.0)
                        if (r < s):
                            v = r/s
                            damp3 = 4*pow(v,3) - 3*pow(v,4);
                            damp5 = pow(v,4);
                            # d/dr derivatives of damping factors
                            damp3_DERIV = 12*pow(v,2) * (1-v) / s
                            damp5_DERIV = 4*pow(v,3) / s

                    fabc = - 3 * damp5_DERIV * pow(r,-6) + 15 * damp5 * pow(r,-7)

                    fc = damp3_DERIV * pow(r,-4) - 3 * damp3 * pow(r,-5)
                    fb = - 3 * damp5 * pow(r,-5)
                    fa = fb
      
                    for a in range(0, 3):
                        for b in range(0, 3):
                            dTab =  fabc * rvec[a] * rvec[b] * rvec[c]
                            if a == b:
                                dTab += fc * rvec[c]
                            if c == a:
                                dTab += fb  * rvec[b]
                            if c == b:
                                dTab += fa  * rvec[a]

                            if k == i:
                                dT[row+a,col+b] = dTab
                            elif k == j:
                                dT[row+a,col+b] = -dTab
                            else:
                                assert k in [i,j], "This is a bug, this case should never be reached!"

        return dT

    def _A_DERIV_POL(self, k, c):
        """
        The derivative of the inverse of   B = (alpha^{-1}E + T)   is obtained as
        
          dA/dx = d(B^{-1})/dx  = - B^{-1} . dB/dx . B^{-1} 
                = - A . dT/dx . A

        since the atomic polarizabilities alpha on the diagonal are constants.
        """
        dT = self._T_DERIV_POL(k, c)
        dA = - np.einsum('im,mn,nj->ij', self.A, dT, self.A)
        return dA

    def _diagA(self):
        """
        diagonal 3x3 blocks of the effective polarizability matrix
        """
        # full (3 npol) x (3 pol) matrix
        diagA = np.zeros((3*self.npol, 3*self.npol))
        # only non-zero elements
        diagA_nonzero = np.zeros((self.npol, 3, 3))
        for k in range(0, self.npol):
            for a in [0,1,2]:
                for b in [0,1,2]:
                    diagA[3*k+a,3*k+b] = self.A[3*k+a,3*k+b]
                    diagA_nonzero[k,a,b] = self.A[3*k+a,3*k+b]
        return diagA, diagA_nonzero

    #####################################################################
    #                                        d Y_{i,j,...}              #
    # CONTRACTIONS   dY/dx (Z) = sum         ------------- Z            #
    #                              i,j,...       d x        i,j,...     #
    #####################################################################
        
    def _gradA(self, U):
        """
        contraction of the gradient of the effective polarizability with a 
        (3 Npol) x (3 Npol) sized matrix U:

           dA              d A_{ij}
           -- (T) = sum    -------- U
           dx          i,j   d x     ij

        This function allows to compute derivatives of functions f(A,...,) as
         
          df/dx = dA/dx . df/dA + ...

        by setting U = df/dA.


        Parameters
        ----------
        U        :   (3*npol, 3*npol) matrix
          partial derivatives df/dA_{i,j} for some function f(A,...)

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information

        Since A only depends on the polarizable atoms, only the middle
        part of `grad` is filled in.
        """
        grad = np.zeros(self.ngrad)

        for ipol in range(0, self.npol):
            for xyz in [0,1,2]:
                dA = self._A_DERIV_POL(ipol,xyz)
                grad[self.startPOL + 3*ipol+xyz] = np.einsum('ij,ij', dA, U)

        return grad

    def _gradDiagA(self, V):
        """
        contraction with the gradients of the diagonal 3x3 blocks of the 
        effective polarizability matrix

           d diag(A)                    d A_{3*k+a,3*k+b}
           --------- (V) = sum  sum     ----------------- V
             d x              k    a,b        d x          k,a,b
         
        Parameters
        ----------
        V        :   (npol, 3, 3) np.ndarray
          partial derivatives df/d(diag A)_{k,a,b} for some function f(diag(A),...)

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information        
        """
        grad = np.zeros(self.ngrad)

        for ipol in range(0, self.npol):
            for xyz in [0,1,2]:
                dA = self._A_DERIV_POL(ipol,xyz)
                grad_ipol = 0.0
                for k in range(0, self.npol):
                    for a in [0,1,2]:
                        for b in [0,1,2]:
                            grad_ipol += dA[3*k+a,3*k+b] * V[k,a,b]
                grad[self.startPOL + 3*ipol+xyz] = grad_ipol

        return grad

    def _gradFnucl(self, N):
        """
        gradient of the contraction of the nuclear fields with a vector N

             (n)               (n)
           dF                dF_i
           ------ (N) = sum  ----- N
            d x            i  d x   i

        Parameters
        ----------
        N      :   (3*npol,) vector
          partial derivatives df/dF^(n)_{i} of some function f(F^(n),...)

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information        
        """
        grad = np.zeros(self.ngrad)

        # gradient on QM atoms
        for inuc in range(0, self.nat):
            for xyz in [0,1,2]:
                dFn = self._F_nucl_DERIV_QM(inuc, xyz)
                grad[self.startQM +3*inuc+xyz] = np.einsum('i,i', dFn, N)
        # gradient on polarizable sites
        for ipol in range(0, self.npol):
            for xyz in [0,1,2]:
                dFn = self._F_nucl_DERIV_POL(ipol, xyz)
                grad[self.startPOL+3*ipol+xyz] = np.einsum('i,i', dFn, N)        
        # gradient on point charges
        for ichg in range(0, self.nchg):
            for xyz in [0,1,2]:
                dFn = self._F_nucl_DERIV_CHG(ichg, xyz)
                grad[self.startCHG+3*ichg+xyz] = np.einsum('i,i', dFn, N)

        return grad

    def _gradFelec(self, E):
        """
        gradient of the contraction of the electronic field integrals with a 
        (3*Npol x Nao x Nao) tensor

             (e)                     (e)
           dF                      dF_{i,m,n}
           ----- (E) = sum  sum    ---------- E
           d x            i    m,n   d x       i,m,n

        Parameters
        ----------
        E      :   (3*npol, nbf, nbf) np.ndarray
          partial derivatives df/dF^(e)_{i,m,n} of some function f(F^(e),...)

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information        
        """
        grad = np.zeros(self.ngrad)
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        for ipol in range(0, self.npol):
            # indices of QM atoms in gradients
            idxQM = np.arange(self.startQM, self.startPOL)
            # indices of polarizable atom i in gradients
            idxPOL = np.arange(self.startPOL+3*ipol, self.startPOL+3*(ipol+1))

            # x-component of 
            #   sum_mn (grad F_mn(Ri)) E_{3*ipol+0,m,n}

            # conversion to a psi4 matrix object
            Eix = psi4.core.Matrix.from_array(E[3*ipol+0,:,:])
            dFEx = self.mints.polarization_integrals_grad(R[ipol,:],
                                                          3, 1,0,0, # k=3, mx=1, my=0, mz=0
                                                          self.cutoff_alpha, self.cutoff_power, Eix).np
            grad[idxQM ] += dFEx.flatten()
            grad[idxPOL] -= np.sum(dFEx, axis=0)
            # y-component
            Eiy = psi4.core.Matrix.from_array(E[3*ipol+1,:,:])
            dFEy = self.mints.polarization_integrals_grad(R[ipol,:],
                                                          3, 0,1,0, # k=3, mx=0, my=1, mz=0
                                                          self.cutoff_alpha, self.cutoff_power, Eiy).np
            grad[idxQM ] += dFEy.flatten()
            grad[idxPOL] -= np.sum(dFEy, axis=0)

            # z-component
            Eiz = psi4.core.Matrix.from_array(E[3*ipol+2,:,:])
            dFEz = self.mints.polarization_integrals_grad(R[ipol,:],
                                                          3, 0,0,1, # k=3, mx=0, my=0, mz=1
                                                          self.cutoff_alpha, self.cutoff_power, Eiz).np
            grad[idxQM ] += dFEz.flatten()
            grad[idxPOL] -= np.sum(dFEz, axis=0)

        return grad

    def _gradI1e(self, Y):
        """
        gradient of the contraction of the exact same-site integrals I^(elec)
        with a (Npol x 3 x 3 x Nao x Nao) tensor Y

           dI                          dI_{k,a,b,m,n}
           -- (E) = sum  sum    sum    -------------  Y
           dx          k    a,b    m,n     d x         k,a,b,m,n

        Parameters
        ----------
        Y      :   (npol, 3, 3, nbf, nbf) np.ndarray
          partial derivatives df/dI_{k,a,b,m,n} of some function f(I^(elec), ...)
          Y is assumed to be symmetric in the 2nd and 3rd indices, Y[:,a,b,:,:]  = Y[:,b,a,:,:]

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information        
        """
        grad = np.zeros(self.ngrad)
        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        def add_gradient(ipol=0, a=0,b=0, mx=2,my=0,mz=0, factor=1):
            """
            add gradient contribution from particular block (ipol,a,b)

                      dI_{ipol,a,b,m,n} 
               sum    ----------------- Y_{ipol,a,b,m,n}
                  m,n      d x
            """
            # indices of QM atoms in gradients
            idxQM = np.arange(self.startQM, self.startPOL)
            # indices of polarizable atom i in gradients
            idxPOL = np.arange(self.startPOL+3*ipol, self.startPOL+3*(ipol+1))

            D = psi4.core.Matrix.from_array( Y[ipol,a,b,:,:] )
            dID = self.mints.polarization_integrals_grad(R[ipol,:],
                                                         6, mx,my,mz,  # k=6
                                                         self.cutoff_alpha, 2*self.cutoff_power, D).np
            grad[idxQM ] += factor * dID.flatten()
            grad[idxPOL] -= factor * np.sum(dID, axis=0)

        for ipol in range(0, self.npol):
            # x^2/r^6
            add_gradient(ipol, 0, 0, mx=2, my=0, mz=0, factor=1)
            # y^2/r^6
            add_gradient(ipol, 1, 1, mx=0, my=2, mz=0, factor=1)
            # z^2/r^6
            add_gradient(ipol, 2, 2, mx=0, my=0, mz=2, factor=1)
            # xy/r^6
            add_gradient(ipol, 0, 1, mx=1, my=1, mz=0, factor=2)
            # xz/r^6
            add_gradient(ipol, 0, 2, mx=1, my=0, mz=1, factor=2)
            # yz/r^6
            add_gradient(ipol, 1, 2, mx=0, my=1, mz=1, factor=2)

        return grad

    def _gradS(self, R):
        """
        gradient of the contraction of the overlap matrix with some generalized density matrix R

          dS              d(S_{m,n})
          -- (R) = sum    ----------  R
          dx          m,n     d x      m,n

        Parameters
        ----------
        R        :   (nbf, nbf) matrix
          partial derivatives df/dS_{m,n} of some function f(S, ...)

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information

        Only gradients on the QM atoms are non-zero.
        """
        grad = np.zeros(self.ngrad)
        # conversion to psi4 matrix object
        R = psi4.core.Matrix.from_array(R)
        # only QM gradients are non-zero
        grad[self.startQM:self.startPOL] = self.mints.overlap_grad(R).np.flatten()

        return grad

    def zero_electron_part_GRAD(self):
        """
        gradient of zero-electron (nuclear/point charges) part of polarization Hamiltonian
        
         d h^(0)
         -------
           dx

        Since nuclei and point charges are kept separately there is also a gradient on 
        the QM atoms.

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information
        """
        K = -np.einsum('ij,j->i', self.A, self.F_nucl)
        L = -0.5*np.einsum('i,j->ij', self.F_nucl, self.F_nucl)
        grad = self._gradFnucl(K) + self._gradA(L)
        return grad

    def one_electron_part_GRAD(self, D):
        """
        contraction of gradient of one-electron integrals with density matrix
            
          dH             d (p|h^(1)|q)
          --(D) = sum    ------------- D
          dx         p,q      d x       p,q

        Parameters
        ----------
        D           :   (nbf, nbf) matrix
          density matrix in AO basis

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information
        """
        if (self.same_site_integrals == 'exact'):
            diagA, diagA_nonzero = self._diagA()
            # only off-diagonal 3x3 blocks
            Abar = self.A-diagA
        else:
            Abar = self.A

        # 1) dH/dA
        FeD = np.einsum('jmn,mn->j', self.F_elec, D)
        FnFeD = np.einsum('i,j->ij', self.F_nucl, FeD)

        FSi = np.einsum('jls,sn->jln', self.F_elec, self.Sinv)
        FSiDt = np.einsum('jln,ml->jmn', FSi, D)
        FFSiDt = np.einsum('imn,jmn->ij', self.F_elec, FSiDt)
        dHdA = -FnFeD - 0.5 * FFSiDt
        # add gradient dA/dx . dH/dA
        grad = self._gradA(dHdA)

        if (self.same_site_integrals == 'exact'):
            # 2) dH/d(diag(A))
            dHdDiagA = np.zeros((self.npol, 3, 3))
            for k in range(0, self.npol):
                for a in [0,1,2]:
                    for b in [0,1,2]:
                        dHdDiagA[k,a,b] = 0.5 * FFSiDt[3*k+a,3*k+b]
            ID = np.einsum('kabmn,mn->kab', self.I_1e, D)
            dHdDiagA -= 0.5 * ID 
            # add gradient d(diagA)/dx . dH/d(diagA)
            grad += self._gradDiagA(dHdDiagA)

        # 3) dH/dF^(nuc)
        dHdFn = -np.einsum('ij,j->i', self.A, FeD)
        # add gradient dF^(nuc)/dx . dH/dF^(nuc)
        grad += self._gradFnucl(dHdFn)

        # 4) dH/dF^(elec)
        AFn = np.einsum('ij,j->i', self.A, self.F_nucl)
        AFnD = np.einsum('i,mn->imn', AFn, D)
        # symmetrize D
        Dsym = 0.5 * (D + np.einsum('ml->lm', D))
        FSiDsym = np.einsum('jln,lm->jmn', FSi, Dsym)
        AbarFSiDsym = np.einsum('ij,jmn->imn', Abar, FSiDsym)
        dHdFe = -AFnD - AbarFSiDsym
        # add gradient dF^(elec)/dx . dH/dF^(elec)
        grad += self._gradFelec(dHdFe)

        if (self.same_site_integrals == 'exact'):
            # 5) dH/dI^(1e)
            dHdI = -0.5 * np.einsum('kab,mn->kabmn', diagA_nonzero, D)
            # add gradient dI^(1)/dx . dH/dI^(1)
            grad += self._gradI1e(dHdI)

        # 6) dH/dS, partial derivatives w/r/t overlap matrix S
        FSiD = np.einsum('igm,gl->ilm', FSi, D)
        AFSi = np.einsum('ij,jln->iln', Abar, FSi)
        dHdS = 0.5 * np.einsum('ilm,iln->mn', FSiD, AFSi)

        # add gradient dS/dx . dH/dS
        grad += self._gradS(dHdS)

        return grad

    def coulomb_J_GRAD(self, D1, D2):
        """
        J-like contraction of gradient of two-electron integrals with two density matrices D1 and D2

          dJ                       (1)  d (pq|h^(2)|rs)   (2)
          --(D1,D2) =  sum        D     ---------------  D
          dx              p,q,r,s  p,q        d x         r,s
           
        Parameters
        ----------
        D1, D2   :   (nbf, nbf) matrices
          density matrices in AO basis

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information
        """
        # dJ/dF
        FD1 = np.einsum('imn,mn->i', self.F_elec, D1)
        FD2 = np.einsum('imn,mn->i', self.F_elec, D2)
        AFD1 = np.einsum('ij,j', self.A, FD1)
        AFD2 = np.einsum('ij,j', self.A, FD2)
        dJdF = -np.einsum('i,mn->imn', AFD2, D1) - np.einsum('i,mn->imn', AFD1, D2)
        # dJ/dA
        dJdA = -np.einsum('i,j->ij', FD1, FD2)
        # evaluate contractions, dJdx = dF/dx . dJ/dF + dA/dx . dJ/dA
        grad = self._gradFelec(dJdF) + self._gradA(dJdA)

        return grad

    def exchange_K_GRAD(self, D1, D2):
        """
        K-like contraction of gradient of two-electron integrals with two density matrices D1 and D2

          dK                       (1)  d (pr|h^(2)|qs)   (2)
          --(D1,D2) =  sum        D     ---------------  D
          dx              p,q,r,s  p,q        d x         r,s
           
        Parameters
        ----------
        D1, D2   :   (nbf, nbf) matrices
          density matrices in AO basis

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information
        """
        FD2 = np.einsum('jls,sn->jln', self.F_elec, D2)
        FD2D1 = np.einsum('jln,lm->jmn', FD2, D1)
        AFD2D1 = np.einsum('ij,jmn->imn', self.A, FD2D1)
        # similarly for the transposes of the density matrices
        FD2t = np.einsum('jls,ns->jln', self.F_elec, D2)
        FD2tD1t = np.einsum('jln,ml->jmn', FD2t, D1)
        AFD2tD1t = np.einsum('ij,jmn->imn', self.A, FD2tD1t)
        # partial derivatives
        dKdF = - (AFD2D1 + AFD2tD1t)
        dKdA = - np.einsum('imn,jmn->ij', self.F_elec, FD2tD1t)
        # gradient contractions, dKdx = dF/dx . dK/dF + dA/dx . dK/dA
        grad = self._gradFelec(dKdF) + self._gradA(dKdA)

        return grad


class RHF_QMMM2ePolGradients(RHF_QMMM2ePol):
    """
    gradients of the RHF energy
    """

    # class for instantiating the polarization Hamiltonian, a static variablee
    PolarizationHamiltonian = PolarizationHamiltonianGradients

    def split_gradient(self, grad):
        """
        split gradients into blocks for QM atoms, polarizable sites and point charges
        """
        nat  = self.molecule.natom()
        npol = self.polarizable_atoms.natom()
        nchg = self.point_charges.natom()
        
        grad_QM  = grad[     :3*nat                ]
        grad_POL = grad[      3*nat:3*(nat+npol)   ]
        grad_CHG = grad[            3*(nat+npol):  ]
        return grad_QM, grad_POL, grad_CHG

    def _electrostatic_embedding_electrons_GRAD(self, point_charges, basis, D):
        """
        contraction of the gradient of the electrostatic potential matrix with a density matrix

                                      (ext)  
          gradVext(D) = sum   { grad  V    }  D
                           ij          ij      ij

        The point charge must have been set with 
        `point_charges.set_nuclear_charge(atom_idx, Qc)`

        Parameters
        ----------
        point_charges   :   psi4.core.Molecule
          external point charges
        basis           :   psi4.core.BasisSet
          basis set with atomic orbitals
        D               :   (nAO, nAO) matrix
          density matrix

        Returns
        -------
        grad_QM   :  (3*nQM,) vector
          gradient on QM atoms
        grad_CHG  :  (3*<num. point charges>,) vector
          gradient on point charges
        """
        # total electron density
        Dt = psi4.core.Matrix.from_array(D)

        grad_QM = np.zeros(3*self.molecule.natom())
        grad_CHG = np.zeros(3*point_charges.natom())
        # The origin of the coordinte system can be shifted to the position
        # of the point charge Rc:
        #
        #   (ext)              Qc                     Q
        #  V      =  sum  <i|------|j>  = <i(r'+Rc)| --- |j(r'+Rc) >    with r' = r-Rc
        #   ij        c      |r-Rc|                   r'
        #
        # Therefore the gradient w/r/t to Rc is equal to (-1) times the sum of the gradientw w/r/t
        # to the centers of the orbitals, Ri and Rj:
        #
        #   d/d(Rc) V   = - d/d(Ri) V   - d/d(Rj) V
        #            ij              ij            ij
        for ichg in range(0, point_charges.natom()):
            epot = core.ExternalPotential()
            epot.addCharge(point_charges.Z(ichg),
                           point_charges.x(ichg),
                           point_charges.y(ichg),
                           point_charges.z(ichg))
            dVdx = epot.computePotentialGradients(self.basis, Dt, False).np
            grad_QM += dVdx.flatten()
            grad_CHG[3*ichg:3*(ichg+1)] -= np.sum(dVdx, axis=0)

        return grad_QM, grad_CHG

    def _electrostatic_embedding_nuclei_GRAD(self, molecule, point_charges):
        """
        gradients of electrostatic energy of nuclei (n=1,...,natom) in the field of the external point charges (c=1,...,<num. point charges>)

                (ext)                  Zn * Qc
          grad E      =  sum sum grad --------
                nuc       n   c        |Rn-Rc|

        The point charge must have been set with 
        `point_charges.set_nuclear_charge(atom_idx, Qc)`

        Parameters
        ----------
        molecule        :   psi4.core.Molecule
          QM region with nuclei
        point_charges   :   psi4.core.Molecule
          external point charges

        Returns
        -------
        grad_QM   :  (3*nQM,) vector
          gradient on QM atoms
        grad_CHG  :  (3*<num. point charges>,) vector
          gradient on point charges        
        """
        grad_QM = np.zeros(3*molecule.natom())
        grad_CHG = np.zeros(3*point_charges.natom())
        if (point_charges.natom() > 0):
            Rnuc = molecule.geometry().np
            Rchrg = point_charges.geometry().np
            # loop over nuclei
            for n in range(0, molecule.natom()):
                Zn = molecule.Z(n)
                # loop over point charges
                for c in range(0, point_charges.natom()):
                    Qc = point_charges.Z(c)
                    rvec = Rnuc[n,:] - Rchrg[c,:]
                    r = la.norm(rvec)

                    grad_QM[3*n:3*(n+1)]  -= Zn*Qc * rvec/pow(r,3)
                    grad_CHG[3*c:3*(c+1)] += Zn*Qc * rvec/pow(r,3)

        return grad_QM, grad_CHG

    def _electrostatic_embedding_charges_GRAD(self, point_charges):
        """
        gradients of electrostatic self-interaction of the external point charges (c,d=1,...,<num. point charges>)

                (ext)                  Qc * Qd
          grad E      =  sum sum grad  -------
                chrg      c  c<d       |Rc-Rd|

        The point charge must have been set with 
        `point_charges.set_nuclear_charge(atom_idx, Qc)`

        Parameters
        ----------
        point_charges   :   psi4.core.Molecule
          external point charges

        Returns
        -------
        grad_CHG  :  (3*<num. point charges>,) vector
          gradient on point charges        
        """
        grad_CHG = np.zeros(3*point_charges.natom())
        if (point_charges.natom() > 0):
            Rchrg = point_charges.geometry().np
            # loop over point charges
            for c in range(0, point_charges.natom()):
                Qc = point_charges.Z(c)
                # loop over point charges
                for d in range(c+1, point_charges.natom()):
                    Qd = point_charges.Z(d)
                    rvec = Rchrg[c,:] - Rchrg[d,:]
                    r = la.norm(rvec)

                    grad_CHG[3*c:3*(c+1)] -= Qc*Qd * rvec/pow(r,3)
                    grad_CHG[3*d:3*(d+1)] += Qc*Qd * rvec/pow(r,3)

        return grad_CHG

    def gradients(self):
        """
        Compute the gradient of the RHF energy

                     core                                               nuc
        d E_RHF   d H             dS          dJ            dK        dV
        ------- = ------- (D)  -  --(W) + 1/2 --(D,D) - 1/4 --(D,D) + --
          d x      d x            dx          dx            dx        dx

        where D is the converged SCF density.

        Returns
        -------
        grad     :  (ngrad,) vector
          gradient w/r/t QM atoms, polarizable sites and point charges,
          see `_gradient_dimensions()` for more information
        """
        # density matrix
        # MO coefficients of doubly occupied orbitals
        Cocc = self.C[:,:self.ndocc]
        # density matrix in AO basis
        D = 2 * np.einsum('ai,bi->ab', Cocc, Cocc)
        # energy-weighted density matrix
        W = 2 * np.einsum('ai,bi,i->ab', Cocc, Cocc, self.e[:self.ndocc])

        # convert to psi4 matrix objects

        # density of spin-up electrons
        Da = psi4.core.Matrix.from_array(0.5*D)
        # density of spin-down electrons
        Db = psi4.core.Matrix.from_array(0.5*D)
        # total electron density
        Dt = psi4.core.Matrix.from_array(D)
        Wt = psi4.core.Matrix.from_array(W)

        # gradient of core Hamiltonian
        dHdx = self.mints.kinetic_grad(Dt).np + self.mints.potential_grad(Dt).np
        dSdx = self.mints.overlap_grad(Wt).np

        # dJ/dx(D,D) and dK/dx(D,D)
        jk_grad = psi4.core.JKGrad.build_JKGrad(1, self.mints)
        jk_grad.set_Ca(psi4.core.Matrix.from_array(Cocc))
        jk_grad.set_Cb(psi4.core.Matrix.from_array(Cocc))
        jk_grad.set_Da(Da)
        jk_grad.set_Db(Db)
        jk_grad.set_Dt(Dt)
        jk_grad.compute_gradient()

        gradients_JK = jk_grad.gradients()
        # psi4's definition of dJ/dx and dK/dx contains a factor of 1/2, i.e.
        #   dJ/dx(D1,D2) = 1/2 sum_{m,n,p,q} D1_{m,n} d( (mn|pq) )/dx D2_{p,q}
        dJdx = 2.0 * gradients_JK["Coulomb"].np
        # I am not sure why we need a factor of 4 instead of 2 here?
        dKdx = 4.0 * gradients_JK["Exchange"].np

        # nuclear gradients
        dEnucDx = self.molecule.nuclear_repulsion_energy_deriv1().np

        # total gradient of plain RHF energy
        dEdx = dEnucDx + dHdx - dSdx + 0.5 * dJdx - 0.25 * dKdx 

        # psi4 gradients have dimension (Nat, 3)
        dEdx = dEdx.flatten()

        # add gradients from polarization energy
        if not self.polham is None:
            # self.polham is an instance of the class `PolarizationHamiltonianGradients` 

            # 0-electron part
            dEnucDx_ = self.polham.zero_electron_part_GRAD()
            # 1-electron part
            dHdx_ = self.polham.one_electron_part_GRAD(D)
            # Coulomb and exchange operators d(\Delta J)/dx(D,D) and d(\Delta K)/dx(D,D)
            dJdx_ = self.polham.coulomb_J_GRAD(D,D)
            dKdx_ = self.polham.exchange_K_GRAD(D,D)

            # gradient of polarization energy
            dEdx_ = dHdx_ + 0.5 * dJdx_ - 0.25 * dKdx_ + dEnucDx_

            # split gradient of polarization energy into different blocks (QM atoms, polarizable sites, point charges)
            grad_QM, grad_POL, grad_CHG = self.split_gradient(dEdx_)
            # and add gradient from plain RHF energy on QM atoms
            grad_QM += dEdx

        else:
            # RHF energy only depends on QM atoms
            grad_QM = dEdx
            grad_POL = np.zeros(3*self.polarizable_atoms.natom())
            grad_CHG = np.zeros(3*self.point_charges.natom())

        # add gradients from electrostatic embedding
        if (self.qmmm == True) and (self.point_charges.natom() > 0):
            # Apparently there is a bug (or at least inconsistency) in psi4's way of computing
            # the external potential due to point charges. It checks the units of the molecule
            # and converts the positions of the point charges to Bohr. However, this conversion
            # seems to have already happened at an earlier stage. Therefore we have to set the units 
            # temporarily to Bohr.
            units = self.point_charges.units()
            self.point_charges.set_units(psi4.core.GeometryUnits.Bohr)

            # gradient from external potential due to MM point charges
            dVextDx_QM, dVextDx_CHG = self._electrostatic_embedding_electrons_GRAD(self.point_charges, self.basis, D)
            grad_QM  += dVextDx_QM
            grad_CHG += dVextDx_CHG

            # revert to previous units if needed
            if (units == "Angstrom"):
                self.point_charges.set_units(psi4.core.GeometryUnits.Angstrom)

            # gradient from electrostatic energy of nuclei in the field of the external 
            # point charges
            dEextDx_QM, dEextDx_CHG = self._electrostatic_embedding_nuclei_GRAD(self.molecule, self.point_charges)
            grad_QM  += dEextDx_QM
            grad_CHG += dEextDx_CHG

            # gradient of the electrostatic interaction among the external point charges themselves 
            dEextDx_CHG = self._electrostatic_embedding_charges_GRAD(self.point_charges)
            grad_CHG += dEextDx_CHG

        # combine gradients into single vector
        grad = np.hstack((grad_QM, grad_POL, grad_CHG))
        return grad

