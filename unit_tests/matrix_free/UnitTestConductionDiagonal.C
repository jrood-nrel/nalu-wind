// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/KokkosViewTypes.h"
#include <Kokkos_Array.hpp>
#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <algorithm>
#include <stk_simd/Simd.hpp>
#include <type_traits>

#include "matrix_free/ConductionDiagonal.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MakeRCP.h"
#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace test_diagonal {
static constexpr int order = 1;
static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
static constexpr int num_elems = 1;

Teuchos::RCP<const Tpetra::Map<>>
make_map()
{
  return make_rcp<Tpetra::Map<>>(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    num_elems * nodes_per_elem, 1,
    make_rcp<Teuchos::MpiComm<int>>(MPI_COMM_WORLD));
}

template <int p>
void
set_aux_fields(elem_offset_view<p> offsets, vector_view<p> coordinates)
{
  constexpr auto nodes = GLL<p>::nodes;
  Kokkos::parallel_for(
    num_elems, KOKKOS_LAMBDA(int index) {
      for (int k = 0; k < p + 1; ++k) {
        const auto cz = nodes[k];
        for (int j = 0; j < p + 1; ++j) {
          const auto cy = nodes[j];
          for (int i = 0; i < p + 1; ++i) {
            const auto cx = nodes[i];
            coordinates(index, k, j, i, 0) = cx;
            coordinates(index, k, j, i, 1) = cy;
            coordinates(index, k, j, i, 2) = cz;
            offsets(index, 0, k, j, i) = 1 + index * simd_len * nodes_per_elem +
                                         k * (order + 1) * (order + 1) +
                                         j * (order + 1) + i;
          }
        }
      }
    });
}

} // namespace test_diagonal
class DiagonalFixture : public ::testing::Test
{
public:
  DiagonalFixture()
  {
    vector_view<order> coordinates{"coordinates", num_elems};
    test_diagonal::set_aux_fields<order>(offsets, coordinates);

    scalar_view<order> alpha{"alpha", num_elems};
    Kokkos::deep_copy(alpha, 1.0);

    scalar_view<order> lambda{"lambda", num_elems};
    Kokkos::deep_copy(lambda, 1.0);

    volume_metric = geom::volume_metric<order>(alpha, coordinates);
    diffusion_metric = geom::diffusion_metric<order>(alpha, coordinates);
  }
  static constexpr int order = 1;
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int num_elems = 1;
  elem_offset_view<order> offsets{"offsets", num_elems};
  scalar_view<order> volume_metric{"volume_metric", num_elems};
  scs_vector_view<order> diffusion_metric{"diffusion_metric", num_elems};
  Tpetra::MultiVector<> rhs{test_diagonal::make_map(), 1};
};

TEST_F(DiagonalFixture, diagonal_executes)
{
  node_offset_view dirichlet_offsets("empty_dirichlet", 1);
  decltype(rhs.getLocalViewDevice()) shared_rhs("empty_rhs", 1, 1);

  rhs.putScalar(0.);
  conduction_diagonal<order>(
    1, offsets, volume_metric, diffusion_metric, rhs.getLocalViewDevice());
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
