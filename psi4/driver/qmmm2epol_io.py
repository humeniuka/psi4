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
The QM/MM-2e-Pol implementation in TeraChem uses the AMBER format for the input files.
In order to compare between the two implementations, this module loads AMBER topology and coordinate
files and converts them to psi4 objects that can be fed to the constructor of the class `PolarizationHamiltonian`.

     TeraChem input files                               psi4 inputs

   topology (prmtop)                        molecule (QM atoms)
   coordinates (rst7)               ---->   point charges (external point charges without nuclei)
   indices of QM atoms (qmregion)           polarizable atoms
                                            atomic polarizabilities
"""
import numpy as np

import psi4
import parmed

def amber2psi4(prmtop_file, rst7_file, qmregion_file):
    """
    This function loads the topology, coordinates and indices of the QM atoms from the three
    files provided as arguments and creates psi4 molecule objects containing the QM atoms, 
    point charges and polarizable sites.
    """
    # Load list of 0-based indices of atoms belonging to the QM region
    qm_indices = np.loadtxt(qmregion_file)
    # load topology and coordinates
    top = parmed.load_file(prmtop_file, xyz=rst7_file)

    param_data = dict( (key, np.array(val)) for (key, val) in top.parm_data.items() )

    if len(param_data['EXCLUDED_ATOMS_LIST']) > 0:
        print("NOTE: Topology file contains non-empty exclusion list, but exclusion lists are not supported yet.")

    # split system into 
    #  * QM atoms
    #  * point charges (without nuclei)
    #  * polarizable atoms
    molecule_str = ""
    polarizable_atoms_str = ""
    point_charges_str = ""
    polarizabilities = {}

    charges = []

    for atom in top.atoms:
        atom_str = f"   {atom.name}   {atom.xx} {atom.xy} {atom.xz}\n"
        
        if atom.idx in qm_indices:
            molecule_str += atom_str
        else:
            pol = param_data["POLARIZABILITY"][atom.idx]
            # Polarizabilities in the AMBER topology file are in Ang^3,
            # we need to convert them to atomic units (Bohr^3).
            BohrToAng = 0.52917724924
            pol *= pow(1.0/BohrToAng, 3)

            if (pol > 0.0):
                polarizable_atoms_str += atom_str
                if atom.name in polarizabilities:
                    assert polarizabilities[atom.name] == pol, "Polarizabilities have to be the same for atoms of the same type."
                polarizabilities[atom.name] = pol
            if (atom.charge != 0.0):
                point_charges_str += atom_str
                charges.append(atom.charge)

    footer = """        

    units angstrom
    no_com
    no_reorient
    symmetry c1
    """

    molecule = psi4.geometry(molecule_str + footer)
    polarizable_atoms = psi4.geometry(polarizable_atoms_str + footer)
    point_charges = psi4.geometry(point_charges_str + footer)
    
    # set correct point charges
    for i in range(0, point_charges.natom()):
        point_charges.set_nuclear_charge(i, charges[i])

    return molecule, polarizable_atoms, point_charges, polarizabilities


