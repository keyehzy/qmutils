cmake_minimum_required(VERSION 3.14)

find_package(OpenMP REQUIRED)

add_executable(
  main
  main.cpp
)

target_link_libraries(
  main
  qmutils
)

target_compile_options(
  main
  PRIVATE
  ${QMUTILS_CXX_COMPILER_OPTIONS}
)

add_library(
  qmutils
  operator.cpp
  term.cpp
  expression.cpp
  normal_order.cpp
  basis.cpp
  matrix_elements.cpp
)

target_include_directories(
  qmutils
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${XXHASHCT_INCLUDE_DIR}
)

target_link_libraries(
  qmutils
  PUBLIC
  xxhashct
  OpenMP::OpenMP_CXX
)

option(USE_BOSON "Build qmutils with bosonic operators" OFF)

if(USE_BOSON)
  target_compile_definitions(
  qmutils
  PUBLIC
  USE_BOSON
  )
endif()
