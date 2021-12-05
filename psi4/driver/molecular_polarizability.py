#!/usr/bin/env python
"""
The polarizability of a molecule is estimated from the atomic polarizabilities of the
constituent atoms and the geometry according to Applequist's and Thole's dipole interaction
model.

References
----------
[Applequist] J. Applequist, J.R. Carl, K-K. Fung
        "An Atom Dipole Interaction Model for Molecular Polarizability.
         Application to Polyatomic Molecules and Determination of Atom Polarizabilities"
         J. Am. Chem. Soc. 94:9 (1972), 2952-2960.
[Thole]  B.T. Thole, 
        "Molecular polarizabilities calculated with a modified dipole interaction."
         Chem. Phys. 59 (1981), 341-350.

"""
import numpy as np
import numpy.linalg as la

# unit conversion factors
bohr2angstrom = 0.52917720859

# atomic dipole polarizabilities in Bohr^3
_theoretical_polarizabilities = {
    # He: theoretical value at FCI/cc-pVDZ level of theory
    "HE" :  0.30274713,
    # add your default polarizabilities for each element below
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

class MolecularPolarizability(object):
    def __init__(self, molecule,
                 polarizabilities=_theoretical_polarizabilities,
                 cutoff_alpha=4.0,
                 dipole_damping='Thole',
                 verbose=1):
        """
        molecular polarizability

        Parameters
        ----------
        molecule            :  psi4.core.Molecule
          molecule whose polarizability should be determined
        polarizabilities    :  dict
          dictionary with atomic polarizabilities for each atom type (in Bohr^3)
        cutoff_alpha        :  float (in Bohr)
          exponent alpha in cutoff function C(r)=(1-exp(-alpha r^2))^q
        dipole_damping      :  str
          The classical dipole polarizability may diverge for certain geometries, unless the dipole-dipole
          interaction is damped at close distances. The following damping functions are available:
            * 'Thole'  - the dipole field tensor is modified according to eqns. A.1 and A.2 in [Thole].
            * 'CPP'    - the same cutoff function as for the CPP integrals is used (see eqn. 4 in [CPP]) (not recommended)
            *  None    - no damping at all  (not recommended)
        verbose             :  int
          controls amount of output written, 0 - silent
        """
        self.verbose = verbose
        self.molecule = molecule
        # all atoms in the molecule are polarizable
        self.npol = self.molecule.natom()
        
        self.polarizabilities = polarizabilities
        self.dipole_damping = dipole_damping
        if (self.verbose > 0):
            print(f"damping of dipole-dipole interaction : {dipole_damping}")

        # cutoff function C(r) = (1- exp(-alpha r^2))^q
        self.cutoff_alpha = cutoff_alpha
        self.cutoff_power = 2

        self.alpha = self._atomic_polarizabilities()
        self.A = self._effective_polarizability()

        # If the polarizable atoms were replaced by a single polarizable site, what
        # would be its polarizability tensor?
        self.alpha_mol = self._molecular_polarizability()

    def _atomic_polarizabilities(self):
        """
        assemble vector of atomic polarizabilities

          alpha = (alpha , ..., alpha , ..., alpha    )
                        1            i            npol

        Returns
        -------
        alpha   :  np.ndarray (Npol,)
          atomic polarizabilities in Bohr^3
        """
        alpha = np.zeros(self.npol)
        for i in range(0, self.npol):
            # atomic dipole polarizability of atom i
            alpha[i] = self.polarizabilities[self.molecule.symbol(i)]
        return alpha
            
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
        R = self.molecule.geometry().np
        
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
                    if self.dipole_damping == 'Thole':
                        r = la.norm(rvec)
                        # see text above eqn. A.1 in [Thole]
                        s = 1.662 * pow(self.alpha[i]*self.alpha[j], 1.0/6.0)
                        if (r < s):
                            v = r/s
                            # damped dipole-dipole interaction according to eqn. A.2 in [Thole]
                            # For v -> 0, T(v) -> 0
                            Tij = 4*pow(v,3)*(1.0-v) * np.eye(3)/pow(r,3) + pow(v,4) * Tij
                    elif self.dipole_damping == 'CPP':
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
        a = alpha_mol * pow(bohr2angstrom, 3)
        if self.verbose > 0:
            print(f""" 

   Total dipole polarizability alpha_mol of molecule (in Ang^3)

               X        Y        Z
        X  {a[0,0]:+8.4f} {a[0,1]:+8.4f} {a[0,2]:+8.4f} 
        Y  {a[1,0]:+8.4f} {a[1,1]:+8.4f} {a[1,2]:+8.4f} 
        Y  {a[2,0]:+8.4f} {a[2,1]:+8.4f} {a[2,2]:+8.4f} 
        """)

        return alpha_mol


if __name__ == "__main__":
    import time
    import psi4
    psi4.set_memory('500 MB')

    # Applequist polarizabilities are taken from table I in [Applequist].
    # Since there are no polarizabilities for aromatic carbon, we take the value for
    # nitrile, as this number is also used by default in AMBER.

    # Numbers are quoted in Angstrom^3 and are converted to Bohr^3.
    polarizabilities = { atom : alpha * pow(1.0/bohr2angstrom, 3) for (atom, alpha) in
                         {
                             "H" : 0.135,     # alkane
                             "C" : 0.360,     # nitrile 
                         }.items()
                     }

    # molecular polarizability of pentacene using Applequist's fitted polarizabilities
    # The geometry was optimized with the AM1 Hamiltonian.
    #
    # In the conjugated pi-system, electrons can flow relatively freely. Some polarization
    # is due to the flow of charge to opposite ends when an external electric field is applied.
    # This effect cannot be captured by Applequist's model. Therefore the polarizability is too low.
    molecule = psi4.geometry("""
    0 1
    C     2.674138     1.792180     0.000000
    C     3.844052     1.101725     0.000000
    C     3.844052    -0.331725     0.000000
    C     2.674138    -1.022180     0.000000
    C     1.406930    -0.335993     0.000000
    C     1.406930     1.105993     0.000000
    C     0.208038     1.790621     0.000000
    C     0.208038    -1.020621    -0.000000
    C    -1.034770    -0.332404    -0.000000
    C    -1.034770     1.102404     0.000000
    C    -2.255323     1.789580     0.000000
    C    -2.255323    -1.019580    -0.000000
    C    -3.475875    -0.332404    -0.000000
    C    -3.475875     1.102404     0.000000
    C    -4.718684     1.790621     0.000000
    C    -4.718684    -1.020621    -0.000000
    C    -5.917576    -0.335993    -0.000000
    C    -5.917576     1.105993    -0.000000
    C    -7.184783     1.792180    -0.000000
    C    -7.184783    -1.022180    -0.000000
    C    -8.354697    -0.331725    -0.000000
    C    -8.354697     1.101725    -0.000000
    H     2.658943     2.892646     0.000000
    H     4.813205     1.622275     0.000000
    H     4.813205    -0.852275     0.000000
    H     2.658943    -2.122646    -0.000000
    H     0.198782     2.891781     0.000000
    H     0.198782    -2.121781    -0.000000
    H    -2.255323     2.890584     0.000000
    H    -2.255323    -2.120584    -0.000000
    H    -4.709427     2.891781     0.000000
    H    -4.709427    -2.121781    -0.000000
    H    -7.169588     2.892646     0.000000
    H    -7.169588    -2.122646    -0.000000
    H    -9.323851    -0.852275    -0.000000
    H    -9.323851     1.622275    -0.000000

    units angstrom
    no_com
    no_reorient
    symmetry c1
    """)

    molpol = MolecularPolarizability(molecule, polarizabilities)

    #print(molpol.alpha_mol)
