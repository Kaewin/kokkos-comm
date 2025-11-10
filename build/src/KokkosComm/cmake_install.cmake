# Install script for directory: /home/research1/repos/kokkos-comm/src/KokkosComm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/KokkosComm" TYPE FILE FILES
    "/home/research1/repos/kokkos-comm/src/KokkosComm/KokkosComm.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/collective.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/concepts.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/fwd.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/point_to_point.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/traits.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/KokkosComm/impl" TYPE FILE FILES "/home/research1/repos/kokkos-comm/src/KokkosComm/impl/contiguous.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/KokkosComm/mpi" TYPE FILE FILES
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/mpi_space.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/comm_mode.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/allgather.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/alltoall.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/barrier.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/channel.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/handle.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/req.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/irecv.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/isend.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/recv.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/send.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/allgather.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/alltoall.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/reduce.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/barrier.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/window.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/KokkosComm/mpi/impl" TYPE FILE FILES
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/impl/pack_traits.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/impl/packer.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/impl/tags.hpp"
    "/home/research1/repos/kokkos-comm/src/KokkosComm/mpi/impl/types.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/KokkosComm" TYPE FILE FILES "/home/research1/repos/kokkos-comm/build/src/KokkosComm/config.hpp")
endif()

