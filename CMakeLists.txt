cmake_minimum_required(VERSION 3.18)
project(HipFalcons LANGUAGES C CXX)

set(FALCONS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Falcons" CACHE PATH "CPU Falcons source tree")
set(FETCHCONTENT_SOURCE_DIR_ZSTDFETCHED "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zstd"
    CACHE PATH "Offline source directory for the bundled Zstd fallback")
set(DTK_ROOT "/opt/dtk" CACHE PATH "DTK or ROCm installation")
set(HIP_ARCH "" CACHE STRING "Optional hipcc offload target, e.g. gfx928")
if(NOT EXISTS "${FALCONS_SOURCE}/include/Falcons/api/falcons.hpp")
    message(FATAL_ERROR
        "Falcons source missing at ${FALCONS_SOURCE}. Use a fresh build directory for bundled dependencies "
        "or set -DFALCONS_SOURCE to a compatible checkout.")
endif()
find_program(HIPCC hipcc HINTS "${DTK_ROOT}/bin" REQUIRED)
find_library(HIP_RUNTIME amdhip64 HINTS "${DTK_ROOT}/lib" "${DTK_ROOT}/lib64" REQUIRED)
set(BUILD_Falcons_BINARY OFF CACHE BOOL "" FORCE)
add_subdirectory("${FALCONS_SOURCE}" cpu-falcons EXCLUDE_FROM_ALL)

if(HIP_ARCH)
    set(arch_flag "--offload-arch=${HIP_ARCH}")
endif()
foreach(unit reconstruct nd_reconstruct)
    set(kernel "${CMAKE_CURRENT_BINARY_DIR}/${unit}.hip.o")
    set(headers "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/stream_params.hpp")
    if(unit STREQUAL "nd_reconstruct")
        list(APPEND headers "${CMAKE_CURRENT_SOURCE_DIR}/nd_plan.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/nd_compact.hpp"
             "${CMAKE_CURRENT_SOURCE_DIR}/dataloader/packed.hpp")
    endif()
    add_custom_command(OUTPUT "${kernel}"
        COMMAND "${HIPCC}" -std=c++17 -O2 -fPIC -ffp-contract=off ${arch_flag}
                -x hip -c "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip.cpp" -o "${kernel}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip.cpp" ${headers} VERBATIM)
    set_source_files_properties("${kernel}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND hip_objects "${kernel}")
endforeach()

add_custom_target(hip_kernels DEPENDS ${hip_objects})
add_executable(hipfalcons main.cpp ${hip_objects})
add_library(hipfalcons_loader SHARED dataloader/codec.cpp ${hip_objects})
add_library(falcons_loader_cpu SHARED dataloader/codec.cpp)
target_compile_definitions(falcons_loader_cpu PRIVATE HF_CPU_ONLY=1)
target_link_libraries(falcons_loader_cpu PRIVATE Falcons::Falcons)
target_include_directories(hipfalcons_loader PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(falcons_loader_cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(TARGET zstd)
    # Falcons' bundled Zstd 1.4.5 keeps zstd_errors.h in lib/common.
    include(FetchContent)
    FetchContent_GetProperties(zstdfetched)
    target_include_directories(falcons_loader_cpu SYSTEM PRIVATE "${zstdfetched_SOURCE_DIR}/lib/common")
endif()
foreach(target hipfalcons hipfalcons_loader)
    add_dependencies(${target} hip_kernels)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__=1)
    target_include_directories(${target} SYSTEM PRIVATE "${DTK_ROOT}/include")
    if(TARGET zstd)
        target_include_directories(${target} SYSTEM PRIVATE "${zstdfetched_SOURCE_DIR}/lib/common")
    endif()
    target_link_libraries(${target} PRIVATE Falcons::Falcons "${HIP_RUNTIME}")
endforeach()

option(HF_BUILD_ENTROPY_TRIAL "Build the experimental HIP entropy verifier" OFF)
if(HF_BUILD_ENTROPY_TRIAL)
    foreach(codec h1 rle escapes)
        set(object "${CMAKE_CURRENT_BINARY_DIR}/entropy_${codec}.hip.o")
        add_custom_command(OUTPUT "${object}"
            COMMAND "${HIPCC}" -std=c++17 -O2 -fPIC ${arch_flag}
                    "-I${FALCONS_SOURCE}/include"
                    -x hip -c "${CMAKE_CURRENT_SOURCE_DIR}/entropy_trial/${codec}.hip.cpp" -o "${object}"
            DEPENDS "entropy_trial/${codec}.hip.cpp" "entropy_trial/${codec}.hpp"
                    "dataloader/h1fast.hpp" "dataloader/rlefast.hpp" VERBATIM)
        set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        list(APPEND entropy_trial_objects "${object}")
    endforeach()
    add_executable(hipfalcons_entropy_trial entropy_trial/main.cpp ${entropy_trial_objects})
    target_compile_features(hipfalcons_entropy_trial PRIVATE cxx_std_17)
    target_compile_definitions(hipfalcons_entropy_trial PRIVATE __HIP_PLATFORM_AMD__=1)
    target_include_directories(hipfalcons_entropy_trial SYSTEM PRIVATE "${DTK_ROOT}/include")
    target_link_libraries(hipfalcons_entropy_trial PRIVATE Falcons::Falcons hipfalcons_loader falcons_loader_cpu "${HIP_RUNTIME}")
    if(TARGET zstd)
        target_include_directories(hipfalcons_entropy_trial SYSTEM PRIVATE "${zstdfetched_SOURCE_DIR}/lib/common")
    endif()
endif()
