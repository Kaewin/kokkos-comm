# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-src"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-build"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src"
  "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/research1/repos/kokkos-comm/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
