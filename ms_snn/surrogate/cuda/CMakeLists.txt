cmake_minimum_required(VERSION 3.18)
project(ms_snn CUDA CXX)

# 自动查找当前目录的源文件
file(GLOB CUDA_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
    message("ARCH_FLAGS = ${ARCH_FLAGS}")
    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()

# 编译选项设置
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        --shared \
        -Xcompiler \
        -fPIC \
        -O3 \
        --use_fast_math \
        --expt-relaxed-constexpr \
        -D_GLIBCXX_USE_CXX11_ABI=0")

# 添加库目标，使用找到的所有源文件
add_library(${PROJECT_NAME} SHARED
    ${CUDA_SOURCES}
)

# # 指定目标架构
# set_property(TARGET  PROPERTY CUDA_ARCHITECTURES 86)

# 设置目标属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA链接
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})