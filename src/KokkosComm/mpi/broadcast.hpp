//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView View>
void broadcast(View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  using Scalar = typename View::value_type;

  if (!KokkosComm::is_contiguous(v)) {
    throw std::runtime_error("low-level broadcast requires contiguous view");
  }

  MPI_Bcast(KokkosComm::data_handle(v), KokkosComm::span(v), KokkosComm::Impl::mpi_type_v<Scalar>, root, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView View>
void broadcast(ExecSpace const& space, View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  space.fence("fence before broadcast");  // work in space may have been used to produce view data
  broadcast(v, root, comm);

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::mpi
