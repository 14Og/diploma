set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_SYSTEM_NAME Generic) 

set(CROSS_COMPILE arm-none-eabi-)
set(CMAKE_C_COMPILER "${CROSS_COMPILE}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_COMPILE}g++")

set(FLAGS_CPU
    "-mthumb"
    "-mcpu=cortex-m4"
    "-march=armv7e-m"
    "-mfpu=fpv4-sp-d16"
    "-mfloat-abi=hard")

string(REPLACE ";" " " FLAGS_CPU "${FLAGS_CPU}")