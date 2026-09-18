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
)
target_include_directories(q38rt PUBLIC include)
target_compile_options(q38rt PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(q38rt PUBLIC dl)

add_executable(q38-runtime src/main.cpp)
target_link_libraries(q38-runtime PRIVATE q38rt)

if(Q38_BUILD_TESTS)
  enable_testing()
  add_executable(q38pack-smoke tests/q38pack_smoke.cpp)
  target_link_libraries(q38pack-smoke PRIVATE q38rt)
  add_test(NAME q38pack-smoke COMMAND q38pack-smoke)
endif()
