#@HEADER
# ************************************************************************
#
#                        Kokkos v. 4.0
#       Copyright (2022) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
# See https://kokkos.org/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#@HEADER


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

####################################################################################

set(KOKKOSCOMM_ENABLE_MPI "ON")
set(KOKKOSCOMM_IMPL_MPI_IS_MPICH "FALSE")
set(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI "TRUE")

GET_FILENAME_COMPONENT(KokkosComm_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)
find_dependency(MPI)
find_dependency(Kokkos)

INCLUDE("${KokkosComm_CMAKE_DIR}/KokkosCommTargets.cmake")
UNSET(KokkosComm_CMAKE_DIR)
