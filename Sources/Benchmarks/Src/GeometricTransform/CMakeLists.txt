set(NAME "geometric_transform")

add_executable(${NAME} GeometricTransformBenchmark.cpp)
target_link_libraries(${NAME} PRIVATE cppbenchmark)
target_include_directories(${NAME} PRIVATE ../)