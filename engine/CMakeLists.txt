cmake_minimum_required(VERSION 3.20)
project(tlm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TLM_SOURCES
  src/model.cpp
  src/tensor_ops.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/transformer.cpp
  src/engine.cpp)

if(EMSCRIPTEN)
  # Two builds of the same engine: simd128 for modern browsers, scalar fallback.
  set(TLM_EXPORTS "_tlm_alloc,_tlm_free,_tlm_load,_tlm_start,_tlm_step,_tlm_piece,_tlm_position,_tlm_seq_len,_tlm_n_params,_tlm_backend")
  foreach(variant scalar simd)
    set(target tlm-${variant})
    add_executable(${target} ${TLM_SOURCES} src/wasm_api.cpp)
    target_include_directories(${target} PRIVATE include)
    target_compile_options(${target} PRIVATE -O3 -fno-exceptions -Wall -Wextra)
    target_link_options(${target} PRIVATE
      -O3
      -sMODULARIZE=1
      -sEXPORT_ES6=1
      -sENVIRONMENT=worker,node
      -sALLOW_MEMORY_GROWTH=1
      -sINITIAL_MEMORY=32MB
      -sMAXIMUM_MEMORY=512MB
      -sFILESYSTEM=0
      -sEXPORTED_FUNCTIONS=${TLM_EXPORTS}
      -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPF32)
    if(variant STREQUAL "simd")
      target_compile_options(${target} PRIVATE -msimd128)
      target_link_options(${target} PRIVATE -msimd128)
    endif()
    set_target_properties(${target} PROPERTIES SUFFIX ".mjs")
  endforeach()
  return()
endif()

option(TLM_NATIVE_ARCH "Compile with -march=native (enables the AVX2 kernel)" ON)

add_library(tlm STATIC ${TLM_SOURCES})
target_include_directories(tlm PUBLIC include)
target_compile_options(tlm PRIVATE -Wall -Wextra)
if(TLM_NATIVE_ARCH)
  target_compile_options(tlm PUBLIC -march=native)
endif()

add_executable(tlm-run src/main.cpp)
target_link_libraries(tlm-run PRIVATE tlm)

add_executable(tlm-bench bench/bench.cpp)
target_link_libraries(tlm-bench PRIVATE tlm)

enable_testing()
add_executable(tlm-tests tests/tests.cpp)
target_link_libraries(tlm-tests PRIVATE tlm)
add_test(NAME unit COMMAND tlm-tests)
