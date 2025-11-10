# CMake generated Testfile for 
# Source directory: /home/research1/repos/kokkos-comm/unit_tests
# Build directory: /home/research1/repos/kokkos-comm/build/unit_tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test-mpi-1]=] "/usr/bin/mpiexec" "-n" "1" "./test-mpi")
set_tests_properties([=[test-mpi-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;68;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-mpi-2]=] "/usr/bin/mpiexec" "-n" "2" "./test-mpi")
set_tests_properties([=[test-mpi-2]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;73;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-gtest-mpi]=] "/usr/bin/mpiexec" "-n" "2" "./test-gtest-mpi")
set_tests_properties([=[test-gtest-mpi]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;143;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-mpi-view-access]=] "/usr/bin/mpiexec" "-n" "2" "./test-mpi-view-access")
set_tests_properties([=[test-mpi-view-access]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;144;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-sendrecv]=] "/usr/bin/mpiexec" "-n" "2" "./test-sendrecv")
set_tests_properties([=[test-sendrecv]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;145;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-isendrecv]=] "/usr/bin/mpiexec" "-n" "2" "./test-isendrecv")
set_tests_properties([=[test-isendrecv]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;146;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-reduce]=] "/usr/bin/mpiexec" "-n" "2" "./test-reduce")
set_tests_properties([=[test-reduce]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;147;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-alltoall]=] "/usr/bin/mpiexec" "-n" "2" "./test-alltoall")
set_tests_properties([=[test-alltoall]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;148;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-allgather]=] "/usr/bin/mpiexec" "-n" "2" "./test-allgather")
set_tests_properties([=[test-allgather]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;149;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-channel]=] "/usr/bin/mpiexec" "-n" "2" "./test-channel")
set_tests_properties([=[test-channel]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;150;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test-window]=] "/usr/bin/mpiexec" "-n" "2" "./test-window")
set_tests_properties([=[test-window]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;151;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
add_test([=[test_lock_unlock]=] "/usr/bin/mpiexec" "-n" "2" "./test_lock_unlock")
set_tests_properties([=[test_lock_unlock]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;136;add_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;152;add_mpi_test;/home/research1/repos/kokkos-comm/unit_tests/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
