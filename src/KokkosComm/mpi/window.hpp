//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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

#include <KokkosComm/traits.hpp>
#include "KokkosComm/mpi/req.hpp"

#include "impl/types.hpp"

namespace KokkosComm {

template <typename ViewType, typename CommSpace = DefaultCommunicationSpace>
class Window {
 public:

  enum class LockType { Shared, Exclusive };

  explicit Window(ViewType v, MPI_Comm comm) : v_(v), comm_(comm) {
    MPI_Win_create(v_.data(), v_.span() * sizeof(typename ViewType::value_type), sizeof(typename ViewType::value_type),
                   MPI_INFO_NULL, comm_, &win);
  }

  ~Window() { MPI_Win_free(&win); }

  void fence (int assert = 0) { MPI_Win_fence(assert, win); }

  /*
  Function: put
  -----------------
  Description:
    Performs an MPI Put operation to write data from the origin address
    to the target rank's memory at the specified displacement.
  */
  template <typename T>
  void put(const T* origin_addr, int count, int target_rank, MPI_Aint target_disp) {
    MPI_Datatype datatype = get_mpi_datatype<T>();

    MPI_Put(origin_addr, count, datatype, target_rank, target_disp, count, datatype, win);
  }

  /*
  Function: get
  -----------------
  Description:
    Performs an MPI Get operation to read data from the source rank's memory
    at the specified displacement into the origin address.
  */
  template <typename T>
  void get(T* origin_addr, int count, int source_rank, MPI_Aint source_disp) {
    MPI_Datatype datatype = get_mpi_datatype<T>();

    MPI_Get(origin_addr, count, datatype, source_rank, source_disp, count, datatype, win);
  }

  /*
  Function: lock
  -----------------
  Description:
    Locks the window for RMA operations on the specified rank with the given lock type.
  */
  void lock(LockType type, int rank, int assert = 0) {
    int mpi_lock_type = (type == LockType::Exclusive) 
                        ? MPI_LOCK_EXCLUSIVE 
                        : MPI_LOCK_SHARED;
    MPI_Win_lock(mpi_lock_type, rank, assert, win);
  }

  /*
  Function: unlock
  -----------------
  Description:
    Unlocks the window for RMA operations on the specified rank.
  */
  void unlock(int rank) { MPI_Win_unlock(rank, win); }

  // NEW - NOT TESTED
  // PSCW Functions:
  void post(MPI_Group post_group, int assert = 0) {
      MPI_Win_post(post_group, assert, win);
  }

  void wait() { MPI_Win_wait(win); }

  void start(MPI_Group start_group, int assert = 0) {
      MPI_Win_start(start_group, assert, win);
  }

  void complete() { MPI_Win_complete(win); }
  // END NEW

 private:
  ViewType v_;
  MPI_Comm comm_;
  MPI_Win win;

  // Helper function to set the datatype for put and get
  template <typename T>
  MPI_Datatype get_mpi_datatype() {
    if (std::is_same<T, int>::value) {
      return MPI_INT;
    } else if (std::is_same<T, int64_t>::value) {
      return MPI_LONG_LONG;
    } else if (std::is_same<T, float>::value) {
      return MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
      return MPI_DOUBLE;
    } else if (std::is_same<T, Kokkos::complex<float>>::value) {
      return MPI_C_FLOAT_COMPLEX;
    } else if (std::is_same<T, Kokkos::complex<double>>::value) {
      return MPI_C_DOUBLE_COMPLEX;
    } else {
      return MPI_BYTE;
    }
  }
};

}  // namespace KokkosComm