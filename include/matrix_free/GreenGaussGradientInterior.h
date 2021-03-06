// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GRADIENT_INTERIOR_H
#define GRADIENT_INTERIOR_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"

#include "Tpetra_MultiVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

namespace impl {
template <int p>
struct gradient_residual_t
{
  static void invoke(
    const const_elem_offset_view<p> offsets,
    const const_scs_vector_view<p> areas,
    const const_scalar_view<p> vols,
    const const_scalar_view<p> q,
    const const_vector_view<p> dqdx_predicted,
    tpetra_view_type yout,
    bool lumped = false);
};
} // namespace impl
P_INVOKEABLE(gradient_residual)

namespace impl {
template <int p>
struct filter_linearized_residual_t
{
  static void invoke(
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> volume_metric,
    ra_tpetra_view_type xin,
    tpetra_view_type yout,
    bool lumped = false);
};
} // namespace impl
P_INVOKEABLE(filter_linearized_residual)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
