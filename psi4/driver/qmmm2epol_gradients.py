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
However, the gradients are computed separately for each group of centers. The type of gradient
computed by a member function is indicated by its name

 * QM atoms           -->    some_quantity_DERIV_QM(k,  c)
 * point charges      -->    some_quantity_DERIV_CHG(k, c)
 * polarizable sites  -->    some_quantity_DERIV_POL(k, c)

`k` is the index of the center in the respective group and `c` enumerates the Cartesian coordinate (x,y or z).
Derivatives are computed for each atom coordinate separately, so that the dimensions of the tensors,
which are differentiated, do not change. 

It is only at the end that the different gradient contributions are added for the same centers.
"""
import numpy as np
import numpy.linalg as la

import psi4
from psi4.driver.qmmm2epol import PolarizationHamiltonian

class PolarizationHamiltonianGradients(PolarizationHamiltonian):
    """
    gradients of two-electron, one-electron and zero-electron parts of the polarization Hamiltonian
    """
    def _monopole_damping(self, r, alpha):
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

        damp, damp_deriv = self._monopole_damping(r, polarizability)

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

    def zero_electron_part_DERIV_QM(self, k, c):
        """
        derivatives of zero-electron Hamiltonian with respect to QM centers
        """
        dF = self._F_nucl_DERIV_QM(k,c)
        # effective polarizability tensor A has no dependency on the QM centers, dA = 0
        # chain rule for product E(x) = -0.5 F(x).A .F(x)
        dE = -0.5 * (
              np.einsum('i,ij,j', dF,          self.A, self.F_nucl)
            + np.einsum('i,ij,j', self.F_nucl, self.A, dF         ))
        return dE

    def zero_electron_part_DERIV_POL(self, k, c):
        """
        derivatives of zero-electron Hamiltonian with respect to polarizable sites
        """
        dF = self._F_nucl_DERIV_POL(k,c)
        dA = self._A_DERIV_POL(k,c)
        # chain rule for product E(x) = -0.5 F(x).A(x).F(x)
        dE = -0.5 * (
              np.einsum('i,ij,j', dF,          self.A, self.F_nucl)
            + np.einsum('i,ij,j', self.F_nucl, dA    , self.F_nucl)
            + np.einsum('i,ij,j', self.F_nucl, self.A, dF         ))
        return dE

    def zero_electron_part_DERIV_CHG(self, k, c):
        """
        derivatives of zero-electron Hamiltonian with respect to nuclei or point charges
        """
        dF = self._F_nucl_DERIV_CHG(k,c)
        # effective polarizability tensor A has no dependency on the point charges, dA = 0
        # chain rule for product E(x) = -0.5 F(x).A .F(x)
        dE = -0.5 * (
              np.einsum('i,ij,j', dF,          self.A, self.F_nucl)
            + np.einsum('i,ij,j', self.F_nucl, self.A, dF         ))
        return dE
        

    def _contracted_gradients_F(self, D):
        """
        compute the gradient of the contraction of the electronic field integrals F^(elec)_{i;m,n} with a
        symmetric density matrix D_{m,n}, that is assumed to be a constant.

        The gradient w/r/t the centers R of the QM atoms is

                                                                  (elec)
          gradFD_QM[3*j:3*(j+1),3*i:3*(i+1)] =  grad      sum    F      (R(1),...,R(nbf); R   (i)) D          i = 1,...,npol ; j = 1,...,nat
                                                    R(j)     m,n  m,n                      pol      m,n

        The gradients on the polarizable site

                                                            (elec)
          gradFD_QM[3*i:3*(i+1),xyz] =  grad        sum    F      (R(1),...,R(nbf); R   (i)) D                  i = 1,...,npol ; xyz = 0,1,2
                                            Rpol(i)    m,n  m,n                      pol      m,n


        can be constructed from gradFD_QM, since 

           dF_{m,n}/dR(pol) = - dF_{m,n}/dR(m) - dF_{m,n}/dR(n)

        where R(m) is the atomic center of basis function m.


        Parameters
        ----------
        D   :  (nbf, nbf) matrix
          symmetric density matrix

        Returns
        -------
        gradFD_QM   :  (3*nat, 3*npol) matrix
          gradient of contraction on QM atoms
        gradFD_POL  :  (3*npol, 3) matrix
          non-zero parts of gradient of contraction on polarizable sites
        """
        # check that D is symmetric, otherwise the gradients will be wrong
        assert np.sum(D - D.T) < 1.0e-10, "Density matrix D has to be symmetric"

        # convert D to a psi4 matrix object
        D = psi4.core.Matrix.from_array(D)

        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        # gradient of F^(elec).D on the QM atoms
        gradFD_QM = np.zeros((3*self.molecule.natom(), 3*self.npol))
        # gradient of F^(elec).D on the polarizable sites
        # Since 
        #        d       (elec)
        #      --------  F       =  0   if  i != j
        #      d Rpol(j)  i,m,n
        # we only have to store the gradients for i == j.
        gradFD_POL = np.zeros((3*self.npol, 3))

        for i in range(0, self.npol):
            # x-component of 
            #   sum_mn (grad F_mn(Ri)) D_mn
            dFxD = self.mints.polarization_integrals_grad(R[i,:],
                                                          3, 1,0,0, # k=3, mx=1, my=0, mz=0
                                                          self.cutoff_alpha, self.cutoff_power, D).np
            gradFD_QM[:,3*i+0] = dFxD.flatten()
            gradFD_POL[3*i:3*(i+1), 0] = -np.sum(dFxD, axis=0)
            # y-component
            dFyD = self.mints.polarization_integrals_grad(R[i,:],
                                                          3, 0,1,0, # k=3, mx=0, my=1, mz=0
                                                          self.cutoff_alpha, self.cutoff_power, D  ).np
            gradFD_QM[:,3*i+1] = dFyD.flatten()
            gradFD_POL[3*i:3*(i+1), 1] = -np.sum(dFyD, axis=0)
            # z-component
            dFzD = self.mints.polarization_integrals_grad(R[i,:],
                                                          3, 0,0,1, # k=3, mx=0, my=0, mz=1
                                                          self.cutoff_alpha, self.cutoff_power, D  ).np
            gradFD_QM[:,3*i+2] = dFzD.flatten()
            gradFD_POL[3*i:3*(i+1), 2] = -np.sum(dFzD, axis=0)

        return gradFD_QM, gradFD_POL

    def _contracted_gradients_I(self, D):
        """
        compute the gradient of the contraction of the electronic field integrals I^(elec)[a,b]_{m,n} with a
        symmetric density matrix D_{m,n}, that is assumed to be a constant.

                             (elec)
           ID[a,b] = sum    I  [a,b] D
                        m,n  m,n      m,n

        Parameters
        ----------
        D   :  (nbf, nbf) matrix
          symmetric density matrix

        Returns
        -------
        gradID_QM   :  (3*nat, npol,3,3) matrix
          gradient of contraction ID on QM atoms
        gradID_POL  :  (3*npol, 3,3) matrix
          non-zero parts of gradient of contraction ID on polarizable sites
        """
        # check that D is symmetric, otherwise the gradients will be wrong
        assert np.sum(D - D.T) < 1.0e-10, "Density matrix D has to be symmetric"

        # convert D to a psi4 matrix object
        D = psi4.core.Matrix.from_array(D)

        # cartesian coordinates of polarizable sites
        R = self.polarizable_atoms.geometry().np

        # gradient of I^(elec) on the QM atoms
        gradID_QM = np.zeros((3*self.molecule.natom(), self.npol, 3,3))
        # gradient of I^(elec) on the polarizable sites
        # We only have to store the gradients for i == j.
        gradID_POL = np.zeros((3*self.npol, 3,3))

        for i in range(0, self.npol):
            # x^2/r^6
            dIxxD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 2,0,0, # k=6, mx=2,my=0,mz=0,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,0,0] = dIxxD.flatten()
            gradID_POL[3*i:3*(i+1), 0,0] = -np.sum(dIxxD, axis=0)
            # y^2/r^6
            dIyyD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 0,2,0, # k=6, mx=0,my=2,mz=0,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,1,1] = dIyyD.flatten()
            gradID_POL[3*i:3*(i+1), 1,1] = -np.sum(dIyyD, axis=0)
            # z^2/r^6
            dIzzD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 0,0,2, # k=6, mx=0,my=0,mz=2,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,2,2] = dIzzD.flatten()
            gradID_POL[3*i:3*(i+1), 2,2] = -np.sum(dIzzD, axis=0)
            # xy/r^6
            dIxyD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 1,1,0, # k=6, mx=1,my=1,mz=0,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,0,1] = dIxyD.flatten()
            gradID_QM[:,i,1,0] = dIxyD.flatten()
            gradID_POL[3*i:3*(i+1), 0,1] = -np.sum(dIxyD, axis=0)
            gradID_POL[3*i:3*(i+1), 1,0] = gradID_POL[3*i:3*(i+1), 0,1]
            # xz/r^6
            dIxzD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 1,0,1, # k=6, mx=1,my=0,mz=1,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,0,2] = dIxzD.flatten()
            gradID_QM[:,i,2,0] = dIxzD.flatten()
            gradID_POL[3*i:3*(i+1), 0,2] = -np.sum(dIxzD, axis=0)
            gradID_POL[3*i:3*(i+1), 2,0] = gradID_POL[3*i:3*(i+1), 0,2]
            # yz/r^6
            dIyzD = self.mints.polarization_integrals_grad(R[i,:],
                                                           6, 0,1,1, # k=6, mx=0,my=1,mz=1,
                                                           self.cutoff_alpha, 2*self.cutoff_power, D).np
            gradID_QM[:,i,1,2] = dIyzD.flatten()
            gradID_QM[:,i,2,1] = dIyzD.flatten()
            gradID_POL[3*i:3*(i+1), 1,2] = -np.sum(dIyzD, axis=0)
            gradID_POL[3*i:3*(i+1), 2,1] = gradID_POL[3*i:3*(i+1), 1,2]

        return gradID_QM, gradID_POL
