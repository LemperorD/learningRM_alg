# CMakeLists.txt: learning_mpc 项目
#
# 项目结构:
#   include/learning_mpc/  → 头文件
#   src/                   → 源文件
#   acados_generated/      → acados Python 脚本生成的 C 求解器代码
#
# 编译前提:
#   1. acados 已编译 (提供 lib/ 和 include/)
#   2. 已运行 python/generate_diff.py 生成 acados_generated/
#
cmake_minimum_required(VERSION 3.10)

# 如果支持，请为 MSVC 编译器启用热重载
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("learning_mpc" LANGUAGES C CXX)

# ================================================================
# acados 路径配置 (根据实际安装位置修改)
# ================================================================
# 通过环境变量 ACADOS_SOURCE_DIR 或在下面手动设置
if(DEFINED ENV{ACADOS_SOURCE_DIR})
    set(ACADOS_PATH $ENV{ACADOS_SOURCE_DIR})
elseif(DEFINED ACADOS_PATH)
    # 已通过 -DACADOS_PATH=... 传入
else()
    # 默认路径 — 请根据实际安装位置修改
    if(WIN32)
        set(ACADOS_PATH "D:/3rdparty/acados")
    else()
        set(ACADOS_PATH "$ENV{HOME}/acados")
    endif()
    message(STATUS "ACADOS_PATH not set, using default: ${ACADOS_PATH}")
endif()

set(ACADOS_INCLUDE_DIR "${ACADOS_PATH}/include")
set(ACADOS_LIB_DIR     "${ACADOS_PATH}/lib")

# ================================================================
# 源文件
# ================================================================
set(SOURCES
    src/main.cpp
    src/acados_mpc_solver.cpp
)

# acados 生成的 C 求解器文件
# 注意: 这些文件由 python/generate_diff.py 生成
# 如果 acados_generated/ 目录不存在，编译会失败 —
# 请先运行: cd python && python generate_diff.py
if(EXISTS "${CMAKE_SOURCE_DIR}/acados_generated")
    file(GLOB ACADOS_GENERATED_SOURCES
        "${CMAKE_SOURCE_DIR}/acados_generated/*.c"
    )
    list(APPEND SOURCES ${ACADOS_GENERATED_SOURCES})
    message(STATUS "Found acados_generated/ with ${ACADOS_GENERATED_SOURCES}")
else()
    message(WARNING "acados_generated/ not found. Run python/generate_diff.py first.")
endif()

# ================================================================
# 可执行文件
# ================================================================
add_executable(learning_mpc ${SOURCES})

# ================================================================
# C++ 标准
# ================================================================
target_compile_features(learning_mpc PUBLIC cxx_std_20)

# ================================================================
# Include 目录
# ================================================================
target_include_directories(learning_mpc PRIVATE
    ${CMAKE_SOURCE_DIR}/include           # learning_mpc/acados_mpc_solver.hpp
    ${ACADOS_INCLUDE_DIR}                 # acados C API headers
    ${ACADOS_INCLUDE_DIR}/acados_c        # ocp_nlp_interface.h 等
    ${ACADOS_INCLUDE_DIR}/blasfeo/include # BLASFEO
    ${ACADOS_INCLUDE_DIR}/hpipm/include   # HPIPM
    ${CMAKE_SOURCE_DIR}/acados_generated  # 生成的 solver 头文件
)

# ================================================================
# 链接 acados 库
# ================================================================
target_link_directories(learning_mpc PRIVATE ${ACADOS_LIB_DIR})

# acados 核心库 (从 link_libs.json 读取实际需要的库)
# 基本: acados, hpipm, blasfeo
find_library(ACADOS_LIB   acados   PATHS ${ACADOS_LIB_DIR} NO_DEFAULT_PATH)
find_library(HPIPM_LIB    hpipm    PATHS ${ACADOS_LIB_DIR} NO_DEFAULT_PATH)
find_library(BLASFEO_LIB  blasfeo  PATHS ${ACADOS_LIB_DIR} NO_DEFAULT_PATH)

if(ACADOS_LIB AND HPIPM_LIB AND BLASFEO_LIB)
    target_link_libraries(learning_mpc PRIVATE
        ${ACADOS_LIB}
        ${HPIPM_LIB}
        ${BLASFEO_LIB}
    )
else()
    message(WARNING "acados libraries not found in ${ACADOS_LIB_DIR}")
    message(WARNING "Build acados first: cd ${ACADOS_PATH} && mkdir build && cd build && cmake .. && make")
endif()

# ================================================================
# 编译器定义
# ================================================================
# ACADOS_MODEL_NAME 宏用于拼接 solver 函数名
target_compile_definitions(learning_mpc PRIVATE ACADOS_MODEL_NAME=differential_drive)

# Windows 特定
if(WIN32)
    target_compile_definitions(learning_mpc PRIVATE _USE_MATH_DEFINES)
endif()

# ================================================================
# 编译选项 (Release 优化)
# ================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(learning_mpc PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>)
endif()
