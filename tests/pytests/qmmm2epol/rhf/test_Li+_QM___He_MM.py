#!/usr/bin/env python
"""
  Li+ (QM) ----- He (MM)

The helium atom (MM) is polarized by the lithium cation (QM)
"""
import pytest
pytestmark = pytest.mark.quick

import psi4
from psi4.driver import qmmm2epol

psi4.set_memory('500 MB')

@pytest.mark.smoke
def test_qmmm2epol():
    #! RHF/cc-pVDZ with QM/MM-2e-pol Hamiltonian for the   Li+ (QM) --- He (MM)   system
    
    #
    # QM atoms and MM atoms as well as additional point charges have to be
    # specified as separate blocks. `no_com` and `no_reorient` prevent
    # the molecules from being rotated or shifted to the center of mass.
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

    # dipole polarizability of helium atom
    polarizabilities = {
        # He: theoretical value at FCI/cc-pVDZ level of theory
        "HE" :  0.30274713
    }
    
    # run closed-shell SCF calculation
    SCF_E = qmmm2epol.rhf_qmmm2epol(molecule, polarizable_atoms, point_charges, 'cc-pVDZ',
                                    polarizabilities=polarizabilities)
    print('Final QMMM-2e-pol SCF energy : %.8f hartree' % SCF_E)
    
    # compare with expected result
    SCF_E_reference = -7.236167358544909
    print('expected                     : %.8f hartree' % SCF_E_reference)
    error_eV = (SCF_E - SCF_E_reference) * 27.211
    print("error = %.8f meV" % (error_eV * 1000))

    assert psi4.compare_values(SCF_E_reference, SCF_E, 6, "SCF QM/MM-2e-pol energy")

