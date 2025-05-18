set(NAME "linear_transform")

add_executable(${NAME} LinearTransformBenchmark.cpp)
target_link_libraries(${NAME} PRIVATE cppbenchmark)
target_include_directories(${NAME} PRIVATE ../)