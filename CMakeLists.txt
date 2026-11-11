cmake_minimum_required(VERSION 3.20)
project(intgrad C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

add_library(intgrad
  src/logquant.c
  src/value.c
  src/net.c
  src/data.c
)
target_include_directories(intgrad PUBLIC include)
target_link_libraries(intgrad PUBLIC m)
target_compile_options(intgrad PRIVATE -Wall -Wextra -Wshadow)

add_executable(ig-train src/train.c)
target_link_libraries(ig-train PRIVATE intgrad)

add_executable(ig-pretty tools/ig-pretty.c)

target_include_directories(ig-pretty PRIVATE include)

enable_testing()
add_executable(test_logquant tests/test_logquant.c)
target_link_libraries(test_logquant PRIVATE intgrad)
add_test(NAME logquant COMMAND test_logquant)

add_executable(test_autograd tests/test_autograd.c)
target_link_libraries(test_autograd PRIVATE intgrad)
add_test(NAME autograd COMMAND test_autograd)
