cmake_minimum_required(VERSION 3.22)

project(diploma_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPPBENCHMARK_MODULE ON)

set(FLAGS_COMMON 
    "-g"
    "-Wall"
    "-Wextra"
    "-Wpedantic"
)

add_subdirectory(CppBenchmark)

string(REPLACE ";" " " FLAGS_COMMON "${FLAGS_COMMON}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_CPU} ${FLAGS_COMMON}")

add_subdirectory(Src/LinearTransform)
add_subdirectory(Src/GeometricTransform)
add_subdirectory(Src/PolynomialTransform)
