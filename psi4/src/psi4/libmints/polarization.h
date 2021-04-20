/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2021 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef _psi_src_lib_libmints_polarization_h_
#define _psi_src_lib_libmints_polarization_h_

#include <vector>
#include "typedefs.h"

#include "psi4/pragma.h"
#include "psi4/libmints/onebody.h"

#include "psi4/libqmmm2epol/polarization.h"

namespace psi {
class SphericalTransform;
class Molecule;

/*! \ingroup MINTS
 *  \class PolarizationInt
 *  \brief Computes multicenter polarization integrals.
 *
 * Use an IntegralFactory to create this object. */
class PolarizationInt : public OneBodyAOInt {

  // position of polarizable atom
  const std::vector<double> &origin;
  // operator    O(r) = x^mx y^my z^mz |r|^-k 
  int k, mx, my, mz;
  // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
  double alpha;
  int q;
  
  //! Computes the polarization integral between two gaussian shells.
  void compute_pair(const GaussianShell &, const GaussianShell &) override;
    
  //! Computes the derivative of the polarization integral between two gaussian shells.
  void compute_pair_deriv1(const GaussianShell &, const GaussianShell &) override;
    
 public:
  //! Constructor. Do not call directly use an IntegralFactory.
  PolarizationInt(std::vector<SphericalTransform> &,
		  std::shared_ptr<BasisSet>, std::shared_ptr<BasisSet>,
		  // position of polarizable atom
		  const std::vector<double> &origin,
		  // operator    O(r) = x^mx y^my z^mz |r|^-k 
		  int k, int mx, int my, int mz,
		  // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		  double alpha,  int q,
		  int deriv = 0);
  
  //! Virtual destructor
  ~PolarizationInt() override;
  
  //! Does the method provide first derivatives?
  bool has_deriv1() override { return true; }
  
};
 
}  // namespace psi

#endif
