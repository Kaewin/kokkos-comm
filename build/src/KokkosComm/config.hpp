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

#define KOKKOSCOMM_VERSION_MAJOR 0
#define KOKKOSCOMM_VERSION_MINOR 2
#define KOKKOSCOMM_VERSION_PATCH 0

#define KOKKOSCOMM_ENABLE_MPI
/* #undef KOKKOSCOMM_IMPL_MPI_IS_MPICH */
#define KOKKOSCOMM_IMPL_MPI_IS_OPENMPI

#if defined(KOKKOSCOMM_ENABLE_MPI) && __has_include(<mpi-ext.h>)
#define KOKKOSCOMM_IMPL_MPIEXT_H
#endif
