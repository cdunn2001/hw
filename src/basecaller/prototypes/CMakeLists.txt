cmake_minimum_required(VERSION 3.20)

project(CudaTemplate LANGUAGES CXX CUDA)

add_subdirectory(common)
add_subdirectory(DataStreamer)
add_subdirectory(BaselineFilter)
add_subdirectory(FrameLabeler)

add_subdirectory(unpacker)
