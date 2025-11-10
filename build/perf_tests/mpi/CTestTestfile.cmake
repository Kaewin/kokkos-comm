# CMake generated Testfile for 
# Source directory: /home/research1/repos/kokkos-comm/perf_tests/mpi
# Build directory: /home/research1/repos/kokkos-comm/build/perf_tests/mpi
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[perf-test-main]=] "/usr/bin/mpiexec" "-n" "2" "./perf-test-main")
set_tests_properties([=[perf-test-main]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/perf_tests/mpi/CMakeLists.txt;33;add_test;/home/research1/repos/kokkos-comm/perf_tests/mpi/CMakeLists.txt;0;")
