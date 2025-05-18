set(NAME "poly_transform")

add_executable(${NAME} PolynomialTransformBenchmark.cpp)
target_link_libraries(${NAME} PRIVATE cppbenchmark)
target_include_directories(${NAME} PRIVATE ../)