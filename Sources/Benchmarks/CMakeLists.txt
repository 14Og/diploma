cmake_minimum_required(VERSION 3.22)

project(diploma_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(Cmake/toolchain.cmake)

set(FLAGS_COMMON 
    "-O2" 
    "-fno-exceptions" 
    "-fno-rtti"
)
set(LINKER_FLAGS
    "-T ${CMAKE_CURRENT_SOURCE_DIR}/Linker/MemoryTemplate.ld"
    "-Wl,--gc-sections"
    "-specs=rdimon.specs"
    "-lc"
    "-lrdimon"
    "-lstdc++"
)

string(REPLACE ";" " " FLAGS_COMMON "${FLAGS_COMMON}")
string(REPLACE ";" " " LINKER_FLAGS "${LINKER_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_CPU} ${FLAGS_COMMON}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_CPU} -O2")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")

file(GLOB_RECURSE HEADER_LIST ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.hpp)

add_executable(${PROJECT_NAME} ${HEADER_LIST} Src/Main.cpp Src/Startup.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE Src)
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".elf")