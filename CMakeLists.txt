cmake_minimum_required(VERSION 3.20)
project(q38rt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Q38_BUILD_TESTS "Build q38rt tests" ON)

add_library(q38rt STATIC
    src/q38pack.cpp
    src/driver.cpp
    src/runtime.cpp
    src/placement.cpp
    src/quant.cpp
)
target_include_directories(q38rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(q38rt PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(q38rt PUBLIC ${CMAKE_DL_LIBS})

add_executable(q38-runtime src/main.cpp)
target_link_libraries(q38-runtime PRIVATE q38rt)

add_executable(q38-gpu-smoke src/gpu_smoke.cpp)
target_link_libraries(q38-gpu-smoke PRIVATE q38rt)

add_executable(q38-plan src/plan.cpp)
target_link_libraries(q38-plan PRIVATE q38rt)

add_executable(q38-rmsnorm-smoke src/rmsnorm_smoke.cpp)
target_link_libraries(q38-rmsnorm-smoke PRIVATE q38rt)

add_executable(q38-q4k-smoke src/q4k_smoke.cpp)
target_link_libraries(q38-q4k-smoke PRIVATE q38rt)

add_executable(q38-q4k-gemv src/q4k_gemv.cpp)
target_link_libraries(q38-q4k-gemv PRIVATE q38rt)

add_executable(q38-q5k-smoke src/q5k_smoke.cpp)
target_link_libraries(q38-q5k-smoke PRIVATE q38rt)

add_executable(q38-q5k-gemv src/q5k_gemv.cpp)
target_link_libraries(q38-q5k-gemv PRIVATE q38rt)

add_executable(q38-q5k-q8k-gemv src/q5k_q8k_gemv.cpp)
target_link_libraries(q38-q5k-q8k-gemv PRIVATE q38rt)

add_executable(q38-q5k-sm86-gemv src/q5k_sm86_gemv.cpp)
target_link_libraries(q38-q5k-sm86-gemv PRIVATE q38rt)

add_executable(q38-iq4xs-smoke src/iq4xs_smoke.cpp)
target_link_libraries(q38-iq4xs-smoke PRIVATE q38rt)

add_executable(q38-iq4xs-gemv src/iq4xs_gemv.cpp)
target_link_libraries(q38-iq4xs-gemv PRIVATE q38rt)

add_executable(q38-iq4xs-sm86-gemv src/iq4xs_sm86_gemv.cpp)
target_link_libraries(q38-iq4xs-sm86-gemv PRIVATE q38rt)

add_executable(q38-iq4xs-q81-gemv src/iq4xs_q81_gemv.cpp)
target_link_libraries(q38-iq4xs-q81-gemv PRIVATE q38rt)

add_executable(q38-layer0-gate src/layer0_gate.cpp)
target_link_libraries(q38-layer0-gate PRIVATE q38rt)

add_executable(q38-layer0-map src/layer0_map.cpp)
target_link_libraries(q38-layer0-map PRIVATE q38rt)

add_executable(q38-layer0-projections src/layer0_projections.cpp)
target_link_libraries(q38-layer0-projections PRIVATE q38rt)

add_executable(q38-gdn-ar-smoke src/gdn_ar_smoke.cpp)
target_link_libraries(q38-gdn-ar-smoke PRIVATE q38rt)

add_executable(q38-recurrent-prep src/recurrent_prep.cpp)
target_link_libraries(q38-recurrent-prep PRIVATE q38rt)

add_executable(q38-layer0-recurrent-front src/layer0_recurrent_front.cpp)
target_link_libraries(q38-layer0-recurrent-front PRIVATE q38rt)

add_executable(q38-layer0-recurrent-attention src/layer0_recurrent_attention.cpp)
target_link_libraries(q38-layer0-recurrent-attention PRIVATE q38rt)

add_executable(q38-layer0-full src/layer0_full.cpp)
target_link_libraries(q38-layer0-full PRIVATE q38rt)

add_executable(q38-layer-topology src/layer_topology.cpp)
target_link_libraries(q38-layer-topology PRIVATE q38rt)

add_executable(q38-recurrent-layer src/recurrent_layer.cpp)
target_link_libraries(q38-recurrent-layer PRIVATE q38rt)

if(Q38_BUILD_TESTS)
  enable_testing()
  add_executable(q38pack-smoke tests/q38pack_smoke.cpp)
  target_link_libraries(q38pack-smoke PRIVATE q38rt)
  add_test(NAME q38pack-smoke COMMAND q38pack-smoke)
  add_executable(q38-placement-smoke tests/placement_smoke.cpp)
  target_link_libraries(q38-placement-smoke PRIVATE q38rt)
  add_test(NAME q38-placement-smoke COMMAND q38-placement-smoke)
endif()
