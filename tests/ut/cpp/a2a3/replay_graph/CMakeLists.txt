# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# replay_graph runtime UTs — mirror of the a2a3 trb runtime UTs (tests/ut/cpp/
# a2a3/*.cpp) but linked against the replay_graph runtime. replay_graph uses the
# run-orch-fully-then-sched model: the orchestrator's submit_task builds the
# fanout dependency graph in-line, and the scheduler only consumes it.
#
# Tests are auto-discovered: drop a `test_<name>.cpp` into this directory and it
# is picked up on the next build as target `test_rg_<name>` — no CMake edit
# needed. GTEST_LIB / GTEST_MAIN_LIB / GTEST_INCLUDE_DIRS are inherited from the
# parent tests/ut/cpp/CMakeLists.txt.
# ---------------------------------------------------------------------------

# Resolve the project src/ root once so the paths below avoid repeated ../../..
# climbs; everything else is derived from these directory variables.
get_filename_component(SRC_ROOT ${CMAKE_SOURCE_DIR}/../../../src ABSOLUTE)
set(REPLAY_GRAPH_DIR ${SRC_ROOT}/a2a3/runtime/replay_graph)
set(REPLAY_GRAPH_RUNTIME_DIR ${REPLAY_GRAPH_DIR}/runtime)

# Single OBJECT lib shared by every replay_graph UT. It bundles the orchestrator
# TU (pto_orchestrator.cpp) and its aicpu deps in addition to the scheduler /
# shared runtime, so auto-discovery needs no per-test variant: the wiring /
# orchestrator-fanin tests need the orchestrator symbols and the rest link them
# harmlessly.
add_library(a2a3_replay_graph_rt_objs OBJECT
    ${REPLAY_GRAPH_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/pto_orchestrator.cpp
    ${SRC_ROOT}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    ${SRC_ROOT}/common/platform/shared/aicpu/tensor_dump_aicpu.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a2a3_replay_graph_rt_objs PUBLIC
    ${REPLAY_GRAPH_DIR}/orchestration
    ${REPLAY_GRAPH_DIR}/runtime
    ${REPLAY_GRAPH_DIR}/common
    ${SRC_ROOT}/a2a3/runtime
    ${SRC_ROOT}/a2a3/platform/include
    ${SRC_ROOT}/common/platform/sim/aicpu
    ${SRC_ROOT}/common/platform/include
    ${SRC_ROOT}/common/task_interface
    ${SRC_ROOT}/common/log/include
    ${SRC_ROOT}/common
)

# Auto-discover every test_*.cpp in this directory. CONFIGURE_DEPENDS re-globs on
# build, so a newly added file is registered without re-running cmake by hand.
file(GLOB RG_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
foreach(test_src ${RG_TEST_SOURCES})
    get_filename_component(test_stem ${test_src} NAME_WE)
    string(REGEX REPLACE "^test_" "test_rg_" target_name ${test_stem})
    add_executable(${target_name} ${test_src})
    target_include_directories(${target_name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE
        a2a3_replay_graph_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES LABELS "no_hardware")
endforeach()
