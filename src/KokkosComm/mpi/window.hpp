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
  explicit Window(ViewType v, MPI_Comm comm) : v_(v), comm_(comm) {
    MPI_Win_create(v_.data(), v_.span() * sizeof(typename ViewType::value_type), sizeof(typename ViewType::value_type),
                   MPI_INFO_NULL, comm_, &win);
  }

  ~Window() { MPI_Win_free(&win); }

  MPI_Win getWin() const { return win; }

  void fence (int assert = 0) { MPI_Win_fence(assert, win); }

  template <typename T>
  void put(const T* origin_addr, int count, int target_rank, MPI_Aint target_disp) {

    MPI_Datatype datatype = get_mpi_datatype<T>();

    MPI_Put(origin_addr, 
            count, 
            datatype, 
            target_rank, 
            target_disp, 
            count, 
            datatype, 
            win);
  }

 private:
  ViewType v_;
  MPI_Comm comm_;
  MPI_Win win;

  // Helper function to set the datatype for put and get
  template <typename T>
  MPI_Datatype get_mpi_datatype() {
    if (std::is_same<T, int>::vaue) {
      return MPI_INT;
    } else if (std::is_same<T, int64_t>::value) {
      return MPI_LONG_LONG;
    } else if (std::is_same<T, float>::value) {
      return MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
      return MPI_DOUBLE;
    } else {
      return MPI_BYTE;
    }
  }
};

}  // namespace KokkosComm
