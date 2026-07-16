cmake_minimum_required(VERSION 3.23)
project(arm4_smoke CXX)
# KC_SRC = path to your kokkos-comm checkout (branch mpi_windows_old_work).
# Headers only. No kokkos-comm configure step, so nothing can hang.
if(NOT DEFINED KC_SRC)
  message(FATAL_ERROR "Pass -DKC_SRC=/path/to/kokkos-comm")
endif()
find_package(Kokkos REQUIRED)   # your Kokkos install (Kokkos_ROOT)
find_package(MPI REQUIRED)
add_executable(smoke_win_view smoke_win_view.cpp)
target_include_directories(smoke_win_view PRIVATE ${KC_SRC}/src ${CMAKE_CURRENT_SOURCE_DIR}/kc_cfg)
target_link_libraries(smoke_win_view PRIVATE Kokkos::kokkos MPI::MPI_CXX)
set_target_properties(smoke_win_view PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(smoke_win_view_dev smoke_win_view_dev.cpp)
target_include_directories(smoke_win_view_dev PRIVATE ${KC_SRC}/src ${CMAKE_CURRENT_SOURCE_DIR}/kc_cfg)
target_link_libraries(smoke_win_view_dev PRIVATE Kokkos::kokkos MPI::MPI_CXX)
set_target_properties(smoke_win_view_dev PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
