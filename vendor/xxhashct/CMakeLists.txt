cmake_minimum_required(VERSION 3.14)

project(xxhashct)
set(XXHASHCT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Include directory for xxhashct")
add_library(xxhashct INTERFACE)
target_include_directories(xxhashct INTERFACE ${XXHASHCT_INCLUDE_DIR})
target_compile_features(xxhashct INTERFACE cxx_std_11)
