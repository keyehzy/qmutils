cmake_minimum_required(VERSION 3.14)

add_executable(
  qmutils-bench
  term-bench.cpp
  expression-bench.cpp
  normal_order-bench.cpp
  basis-bench.cpp
  sparse_matrix-bench.cpp
  lru_cache-bench.cpp
)

target_include_directories(
  qmutils-bench
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  qmutils-bench
  PRIVATE
  benchmark::benchmark_main
  pthread
  qmutils
)
