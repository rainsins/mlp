cmake_minimum_required(VERSION 3.10)
project(MLP_MNIST)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
# 开启最高等级优化，加速 CPU 计算
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3 -march=native")

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(mlp_trainer ${SOURCES})

# add_executable(mlp_gpu ${SOURCES})