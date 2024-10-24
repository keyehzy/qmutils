cmake_minimum_required(VERSION 3.14)

find_package(Armadillo REQUIRED)

function(add_example example_file)
  get_filename_component(
    example_name
    ${example_file}
    NAME_WE
  )

  add_executable(
    ${example_name}
    ${example_file}
  )

  target_compile_options(
    ${example_name}
    PRIVATE
    ${QMUTILS_CXX_COMPILER_OPTIONS}
  )

  target_include_directories(
    ${example_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XXHASHCT_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${example_name}
    PRIVATE
    qmutils
    xxhashct
    ${ARMADILLO_LIBRARIES}
  )

endfunction()

file(GLOB EXAMPLE_FILES
  hubbard_1d.cpp
  hubbard_1d_mean_field.cpp
  fourier_transform_1d.cpp
  fourier_transform_2d.cpp
  ssh_model.cpp
  frustrated_kagome.cpp
  heisenberg_1d.cpp
  heisenberg_2d.cpp
)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  add_example(${EXAMPLE_FILE})
endforeach()
