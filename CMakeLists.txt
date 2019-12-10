cmake_minimum_required(VERSION 3.15)
project(ch7new)

set(CMAKE_CXX_STANDARD 11)


add_executable(ch7new main.cpp)

find_package(OpenCV 3.1 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

target_link_libraries(ch7new ${CERES_LIBRARIES} ${OpenCV_LIBS} ${Sophus_LIBS})