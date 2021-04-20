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
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include <stdexcept>

#include "psi4/libmints/polarization.h"

// for DEBUG
#include <iostream>

using namespace psi;

PolarizationInt::PolarizationInt(std::vector<SphericalTransform> &spherical_transforms,
				 std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2,
				 // position of polarizable atom
				 const std::vector<double> &origin,
				 // operator    O(r) = x^mx y^my z^mz |r|^-k 
				 int k, int mx, int my, int mz,
				 // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
				 double alpha,  int q,
				 int deriv)
  : OneBodyAOInt(spherical_transforms, bs1, bs2, deriv),
    origin{origin},
    k{k}, mx{mx}, my{my}, mz{mz},
    alpha{alpha}, q{q}
{

    int maxam1 = bs1_->max_am();
    int maxam2 = bs2_->max_am();

    int maxnao1 = INT_NCART(maxam1);
    int maxnao2 = INT_NCART(maxam2);

    if (deriv == 0) {
      set_chunks(1);
      buffer_ = new double[maxnao1 * maxnao2];
    } else if (deriv == 1) {
      // derivatives with respect to
      //  (1) center of bra orbital
      //  (2) center of polarizable site
      //  (3) center of ket orbital
      set_chunks(9);
      buffer_ = new double[9 * maxnao1 * maxnao2];
    } else  {
      throw std::runtime_error("PolarizationInt: does not support 2nd order derivatives and higher.");
    }
}

PolarizationInt::~PolarizationInt() {
  delete[] buffer_;
}


// The engine only supports segmented basis sets
void PolarizationInt::compute_pair(// shells of orbitals in bra and ket
				   const GaussianShell &s1, const GaussianShell &s2) {
  int ao12;
  double A[3], B[3];
  int i;
  // shift polarizable site to origin of coordinate system
  for(i=0; i<3; i++) {
    A[i] = s1.center()[i] - origin[i];
    B[i] = s2.center()[i] - origin[i];
  }
  // angular momenta of the shells
  int am1 = s1.am();
  int am2 = s2.am();
  // number of primitives
  int nprim1 = s1.nprimitive();
  int nprim2 = s2.nprimitive();

  memset(buffer_, 0, s1.ncartesian() * s2.ncartesian() * sizeof(double));

  for (int p1 = 0; p1 < nprim1; ++p1) {
    double a1 = s1.exp(p1);
    double c1 = s1.coef(p1);
    for (int p2 = 0; p2 < nprim2; ++p2) {
      double a2 = s2.exp(p2);
      double c2 = s2.coef(p2);

      PolarizationIntegral integrals(A[0],A[1],A[2], am1, a1,
				     B[0],B[1],B[2], am2, a2,
				     k, mx,my,mz,
				     alpha, q);
      
      ao12 = 0;
      /*--- create all am components (l1,m1,n1) of si ---*/
      for (int ii = 0; ii <= am1; ii++) {
	int l1 = am1 - ii;
	for (int jj = 0; jj <= ii; jj++) {
	  int m1 = ii - jj;
	  int n1 = jj;
	  /*--- create all am components (l2,m2,n2) of sj ---*/
	  for (int kk = 0; kk <= am2; kk++) {
	    int l2 = am2 - kk;
	    for (int ll = 0; ll <= kk; ll++) {
	      int m2 = kk - ll;
	      int n2 = ll;

	      buffer_[ao12++] += c1 * c2 * integrals.compute_pair(l1,m1,n1, l2,m2,n2);
	    }
	  }
	}
      }
    }
  }
}

// The engine only supports segmented basis sets
void PolarizationInt::compute_pair_deriv1(// shells of orbitals in bra and ket
					  const GaussianShell &s1, const GaussianShell &s2) {

}
