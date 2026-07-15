/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Runtime Builder - rt2 Implementation (Device Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * Supports device orchestration where AICPU thread 3 runs the orchestrator.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - Copies orchestration SO to device memory
 *   - Sets up runtime state for device orchestration
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Frees device memory
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/runtime.h"
#include "../../../../common/task_interface/call_config.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/platform_compile_info.h"
#include "host/raii_scope_guard.h"
#include "common/host_api.h"
#include "host_graph.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

static_assert(RUNTIME_ENV_RING_COUNT >= 1, "RuntimeEnv must provide the replay-graph sizing slot");

// Helper: return current time in milliseconds
static int64_t _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

static int64_t _steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

extern "C" uint64_t host_graph_record_now_ns() { return static_cast<uint64_t>(_steady_now_ns()); }

extern "C" uint64_t host_graph_record_minor_faults() {
    struct rusage usage {};
#ifdef RUSAGE_THREAD
    const int who = RUSAGE_THREAD;
#else
    const int who = RUSAGE_SELF;
#endif
    return getrusage(who, &usage) == 0 ? static_cast<uint64_t>(usage.ru_minflt) : 0;
}

static bool is_power_of_2_u64(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

static bool replay_graph_host_orch_enabled() {
    const char *value = std::getenv("PTO_REPLAY_GRAPH_HOST_ORCH");
    if (value == nullptr) return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
}

static bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t written = 0;
    while (written < size) {
        ssize_t rc = write(fd, data + written, size - written);
        if (rc <= 0) return false;
        written += static_cast<size_t>(rc);
    }
    return true;
}

static bool create_orch_so_tempfile(const uint8_t *data, size_t size, std::string *out_path) {
    char path[] = "/tmp/replay_graph_host_orch_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return false;

    bool ok = fchmod(fd, 0755) == 0 && write_all_bytes(fd, data, size);
    if (close(fd) != 0) ok = false;
    if (!ok) {
        unlink(path);
        return false;
    }
    *out_path = path;
    return true;
}

using HostOrchestrationEntryFunc = void (*)(const L2TaskArgs &);
using HostOrchestrationBindRuntimeFunc = void (*)(PTO2Runtime *);

struct HostOrchEntryPoints {
    HostOrchestrationEntryFunc entry{nullptr};
    HostOrchestrationBindRuntimeFunc bind{nullptr};
};

static void destroy_host_orch_entry_points(void *ptr) { delete static_cast<HostOrchEntryPoints *>(ptr); }

static std::string trim_copy(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

static bool parse_uint_token(
    const char *name, const std::string &raw, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t *out
) {
    std::string token = trim_copy(raw);
    if (token.empty()) {
        LOG_WARN("%s has an empty value in '%s', ignored", name, raw.c_str());
        return false;
    }

    if (token[0] == '-') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    char *endptr = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == token.c_str() || *endptr != '\0') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    uint64_t val = static_cast<uint64_t>(parsed);

    if (val < min_val || val > max_val) {
        LOG_WARN(
            "%s=%s invalid (must be in [%" PRIu64 ", %" PRIu64 "]), ignored", name, token.c_str(), min_val, max_val
        );
        return false;
    }
    if (require_power_of_2 && !is_power_of_2_u64(val)) {
        LOG_WARN("%s=%s invalid (must be a power of 2), ignored", name, token.c_str());
        return false;
    }
    *out = val;
    return true;
}

static void
apply_env_value(const char *name, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t *out) {
    const char *env = std::getenv(name);
    if (!env) return;
    std::string text(env);
    if (text.find(',') != std::string::npos) {
        LOG_WARN("%s=%s invalid for single-ring replay_graph (expected one value), ignored", name, env);
        return;
    }
    (void)parse_uint_token(name, text, min_val, max_val, require_power_of_2, out);
}

// ring_task_window / ring_heap / ring_dep_pool point into the #pragma pack(1)
// RuntimeEnv wire struct (call_config.h), so their uint64_t entries are only
// byte-aligned — runtime_env sits at offset 32 in CallConfig (after 8 int32_t),
// which is not guaranteed to satisfy uint64_t alignment under packed ABI.
// Reading them as `base[idx]` is an
// unaligned 8-byte load: UB, and fatal under UBSan (-fsanitize=alignment). Copy
// the bytes out instead. A null base means "no per-task overrides" -> 0 (unset).
static uint64_t read_ring_override(const uint64_t *base, int idx) {
    if (base == nullptr) {
        return 0;
    }
    uint64_t value;
    std::memcpy(&value, base + idx, sizeof(value));
    return value;
}

// The public RuntimeEnv ABI still carries arrays. replay_graph consumes slot 0
// as a whole-graph override and deliberately ignores the legacy ring slots.
static bool resolve_graph_config(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool,
    uint64_t *task_window_size, uint64_t *heap_size, int32_t *dep_pool_capacity
) {
    *task_window_size = PTO2_TASK_WINDOW_SIZE;
    *heap_size = PTO2_HEAP_SIZE;
    uint64_t dep_pool_value = PTO2_DEP_LIST_POOL_SIZE;
    apply_env_value("PTO2_RING_TASK_WINDOW", 4, static_cast<uint64_t>(INT32_MAX), true, task_window_size);
    apply_env_value("PTO2_RING_HEAP", 1024, std::numeric_limits<uint64_t>::max(), false, heap_size);
    apply_env_value("PTO2_RING_DEP_POOL", 4, static_cast<uint64_t>(INT32_MAX), false, &dep_pool_value);

    uint64_t override_value = read_ring_override(ring_task_window, 0);
    if (override_value != 0) *task_window_size = override_value;
    override_value = read_ring_override(ring_heap, 0);
    if (override_value != 0) *heap_size = override_value;
    override_value = read_ring_override(ring_dep_pool, 0);
    if (override_value != 0) dep_pool_value = override_value;

    if (*task_window_size < 4 || *task_window_size > static_cast<uint64_t>(INT32_MAX) ||
        !is_power_of_2_u64(*task_window_size)) {
        LOG_ERROR("task_window=%" PRIu64 " must be a power of 2 in [4, INT32_MAX]", *task_window_size);
        return false;
    }
    if (*heap_size < 1024) {
        LOG_ERROR("heap=%" PRIu64 " must be >= 1024", *heap_size);
        return false;
    }
    if (dep_pool_value < 4 || dep_pool_value > static_cast<uint64_t>(INT32_MAX)) {
        LOG_ERROR("dep_pool=%" PRIu64 " must be in [4, INT32_MAX]", dep_pool_value);
        return false;
    }
    *dep_pool_capacity = static_cast<int32_t>(dep_pool_value);
    return true;
}

static int32_t pto2_read_runtime_status(Runtime *runtime, const HostApi *api, PTO2SharedMemoryHeader *host_header) {
    if (runtime == nullptr || host_header == nullptr) {
        return 0;
    }

    void *pto2_sm = runtime->get_gm_sm_ptr();
    if (pto2_sm == nullptr) {
        return 0;
    }

    int hdr_rc = api->copy_from_device(host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
    if (hdr_rc != 0) {
        LOG_WARN("Failed to copy PTO2 header from device");
        return 0;
    }

    int32_t orch_error_code = host_header->orch_error_code.load(std::memory_order_relaxed);
    int32_t sched_error_code = host_header->sched_error_code.load(std::memory_order_relaxed);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static void release_tensor_leases(Runtime *runtime, const HostApi *api) {
    int freed = 0;
    int buffer_noop = 0;
    int external_noop = 0;
    for (TensorLease &lease : runtime->tensor_leases_) {
        if (lease.dev_ptr == nullptr) {
            continue;
        }
        if (lease.host_mapped && api->unregister_device_memory_from_host != nullptr) {
            api->unregister_device_memory_from_host(lease.dev_ptr);
            lease.host_mapped = false;
        }
        switch (lease.release_kind) {
        case TensorReleaseKind::Free:
            api->device_free(lease.dev_ptr);
            ++freed;
            break;
        case TensorReleaseKind::BufferNoop:
            ++buffer_noop;
            break;
        case TensorReleaseKind::ExternalNoop:
            ++external_noop;
            break;
        }
    }
    LOG_DEBUG("Released tensor leases: freed=%d buffer_noop=%d external_noop=%d", freed, buffer_noop, external_noop);
    runtime->tensor_leases_.clear();
}

// per-run bump allocator over the runner's retained temporary buffer. This is
// the whole temporary-buffer mechanism: the platform only remembers a
// {addr, size} slot across runs (HostApi get/set_retained_temp_buffer); the
// grow/pack/slice logic lives here. TRB kernels require 1024-byte-aligned
// device pointers, which device_malloc already guarantees for the OFF path, so
// the retained base is 1024-aligned and slices taken at 1024-aligned offsets
// stay aligned without any base fix-up.
class RetainedTempBump {
public:
    static constexpr size_t kAlignment = 1024;

    static size_t align_up(size_t v) { return (v + (kAlignment - 1)) & ~(kAlignment - 1); }

    // Pack the run's non-child, non-empty tensors to compute the required
    // aligned size, then grow the retained slot if it is too small (free old +
    // malloc new + write back). Returns false only if the (grow) device_malloc
    // fails. A run needing 0 bytes leaves the slot untouched.
    bool begin(const HostApi *api, const ChipStorageTaskArgs *orch_args) {
        api_ = api;
        offset_ = 0;
        size_t required = 0;
        for (int i = 0; i < orch_args->tensor_count(); i++) {
            Tensor t = orch_args->tensor(i);
            if (t.is_child_memory() || t.nbytes() == 0) {
                continue;
            }
            required += align_up(static_cast<size_t>(t.nbytes()));
        }
        void *addr = nullptr;
        size_t size = 0;
        api->get_retained_temp_buffer(&addr, &size);
        if (required > size) {
            if (addr != nullptr) {
                api->device_free(addr);
            }
            addr = required != 0 ? api->device_malloc(required) : nullptr;
            if (required != 0 && addr == nullptr) {
                api->set_retained_temp_buffer(nullptr, 0);
                base_ = nullptr;
                capacity_ = 0;
                LOG_ERROR("Retained temp buffer grow failed: required bytes %zu", required);
                return false;
            }
            api->set_retained_temp_buffer(addr, required);
            size = required;
        }
        base_ = addr;
        capacity_ = size;
        return true;
    }

    // Slice `bytes` from the retained buffer at the next 1024-aligned offset.
    // Must fit because begin() sized the buffer from the same tensors; a miss
    // is a caller bug (plan/slice mismatch), reported as nullptr.
    void *acquire(size_t bytes) {
        size_t aligned = align_up(offset_);
        if (base_ == nullptr || aligned + bytes > capacity_) {
            LOG_ERROR("Retained temp buffer slice miss: bytes=%zu offset=%zu capacity=%zu", bytes, aligned, capacity_);
            return nullptr;
        }
        void *ptr = static_cast<char *>(base_) + aligned;
        offset_ = aligned + bytes;
        return ptr;
    }

private:
    const HostApi *api_ = nullptr;
    void *base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * the supplied runtime so a subsequent bind_callable_to_runtime_impl can use
 * them. This is the cacheable half of init_runtime_impl: nothing here depends
 * on per-run argument values, so the simpler_register_callable / simpler_run split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @return 0 on success, -1 on failure
 */
extern "C" int
register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }
    if (upload_fn == nullptr || out == nullptr) {
        LOG_ERROR("upload_fn or out is null");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO_V0("Registering %d kernel(s) in register_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(callable, upload_fn, &out->kernel_addrs) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            return -1;
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for device orchestration");
        return -1;
    }

    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();

    if (replay_graph_host_orch_enabled()) {
        const char *entry_name = callable->func_name();
        if (entry_name == nullptr || entry_name[0] == '\0') {
            LOG_ERROR("replay_graph host orchestration requires a non-empty entry symbol");
            return -1;
        }

        std::string so_path;
        if (!create_orch_so_tempfile(orch_so_binary, orch_so_size, &so_path)) {
            LOG_ERROR("Failed to materialize replay_graph orchestration SO for host loading");
            return -1;
        }

        dlerror();
        void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        const char *load_error = dlerror();
        if (handle == nullptr) {
            LOG_ERROR("Host orchestration dlopen failed: %s", load_error ? load_error : "unknown error");
            unlink(so_path.c_str());
            return -1;
        }

        dlerror();
        auto entry = reinterpret_cast<HostOrchestrationEntryFunc>(dlsym(handle, entry_name));
        const char *entry_error = dlerror();
        if (entry == nullptr || entry_error != nullptr) {
            LOG_ERROR(
                "Host orchestration dlsym('%s') failed: %s", entry_name,
                entry_error ? entry_error : "null entry"
            );
            dlclose(handle);
            unlink(so_path.c_str());
            return -1;
        }

        dlerror();
        auto bind = reinterpret_cast<HostOrchestrationBindRuntimeFunc>(dlsym(handle, "framework_bind_runtime"));
        const char *bind_error = dlerror();
        if (bind == nullptr || bind_error != nullptr) {
            LOG_ERROR(
                "Host orchestration dlsym('framework_bind_runtime') failed: %s",
                bind_error ? bind_error : "null entry"
            );
            dlclose(handle);
            unlink(so_path.c_str());
            return -1;
        }

        unlink(so_path.c_str());
        auto *entry_points = new HostOrchEntryPoints{entry, bind};
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = entry_points;
        out->host_orch_func_ptr_deleter = destroy_host_orch_entry_points;
        LOG_INFO_V0("Loaded replay_graph orchestration entry '%s' on host", entry_name);
    }
    LOG_INFO_V0("Orchestration SO: %zu bytes staged (host-only)", orch_so_size);
    return 0;
}

// Effective graph sizing for one (callable_id, config): the input half of the
// arena description. Resolved once per config from per-task overrides + env +
// compile-time defaults; depends on nothing that varies per run.
struct ArenaSizingConfig {
    uint64_t task_window_size;
    uint64_t heap_size;
    int32_t dep_pool_capacity;
};

struct ArenaStaticSizes {
    uint64_t heap_size;
    uint64_t sm_size;
};

// Device pointers to the per-Worker static pools that DeviceRunner keeps alive
// across runs (freed in DeviceRunner::finalize(), never in tensor_leases_).
struct StaticArenaPtrs {
    void *gm_heap;
    void *gm_sm;
    void *runtime_arena_dev;
};

struct PrebuiltRuntimeArenaCacheProbe {
    uint64_t hash{0};
    std::vector<uint8_t> serialized_key{};
};

static void hash_mix_u64(uint64_t *hash, uint64_t value) {
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    for (int i = 0; i < 8; i++) {
        *hash ^= (value >> (i * 8)) & 0xff;
        *hash *= kFnvPrime;
    }
}

static void append_cache_key_u64(std::vector<uint8_t> *out, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        out->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
}

static PrebuiltRuntimeArenaCacheProbe make_prebuilt_runtime_arena_cache_probe(const ArenaSizingConfig &sizing) {
    PrebuiltRuntimeArenaCacheProbe probe;
    uint64_t hash = 1469598103934665603ULL;
    probe.serialized_key.reserve(3 * sizeof(uint64_t));
    hash_mix_u64(&hash, sizing.task_window_size);
    append_cache_key_u64(&probe.serialized_key, sizing.task_window_size);
    hash_mix_u64(&hash, sizing.heap_size);
    append_cache_key_u64(&probe.serialized_key, sizing.heap_size);
    hash_mix_u64(&hash, static_cast<uint32_t>(sizing.dep_pool_capacity));
    append_cache_key_u64(&probe.serialized_key, static_cast<uint32_t>(sizing.dep_pool_capacity));
    probe.hash = hash;
    return probe;
}

// per-(cid,config): resolve the cache-key sizing knobs. Pure host parsing over
// per-task overrides, PTO2_RING_* env, and compile-time defaults. Derived
// allocation sizes are computed only on cache miss.
static bool resolve_arena_sizing(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool, ArenaSizingConfig *out
) {
    if (!resolve_graph_config(
            ring_task_window, ring_heap, ring_dep_pool, &out->task_window_size, &out->heap_size, &out->dep_pool_capacity
        )) {
        return false;
    }
    LOG_INFO_V0(
        "Replay graph sizes: task_window=%" PRIu64 " heap=%" PRIu64 " dep_pool=%d", out->task_window_size,
        out->heap_size, out->dep_pool_capacity
    );

    return true;
}

static bool derive_arena_static_sizes(const ArenaSizingConfig &sizing, ArenaStaticSizes *out) {
    out->heap_size = sizing.heap_size;
    out->sm_size = PTO2SharedMemoryHandle::calculate_size(sizing.task_window_size);
    return true;
}

// per-run: the only signature-aware step. Copy the orch args, replacing each
// host tensor pointer with a freshly staged device pointer (H2D copy-in, or an
// on-device zero for pure-OUTPUT buffers), and record the host/device pair for
// copy-back. Read-only INPUT tensors skip copy-back. When `bump` is non-null,
// ordinary non-child tensors are sliced from the runner's retained temporary
// buffer (released as a no-op — the buffer is reused across runs); otherwise
// each is device_malloc'd and freed in validate. On failure the partially
// staged device_args / tensor_leases_ stay owned by the caller's Runtime.
static bool stage_device_args(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, const ArgDirection *signature,
    int sig_count, RetainedTempBump *bump, bool map_for_host_orch, ChipStorageTaskArgs *out
) {
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();

    int64_t t_args_start = _now_ms();
    STRACE_A("simpler_run.bind.args", "");
    for (int i = 0; i < tensor_count; i++) {
        Tensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            out->add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());
        if (size == 0) {
            t.buffer.addr = 0;
            out->add_tensor(t);
            continue;
        }

        void *dev_ptr = nullptr;
        TensorReleaseKind release_kind = TensorReleaseKind::Free;
        if (bump != nullptr) {
            dev_ptr = bump->acquire(size);
            release_kind = TensorReleaseKind::BufferNoop;
            if (dev_ptr == nullptr) {
                LOG_ERROR("Retained temp buffer slice failed for tensor %d: tensor bytes=%zu", i, size);
                return false;
            }
        } else {
            dev_ptr = api->device_malloc(size);
        }
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return false;
        }

        // Pure write-only OUTPUT buffers carry no meaningful host content, so
        // the H2D copy-in is wasted. Zero them on-device instead (cheap HBM
        // memset, no PCIe) so any region the kernel leaves unwritten reads as 0
        // rather than pooled-allocator garbage. INOUT (read-before-write)
        // and IN keep the H2D copy. Falls back to copy_to_device if a backend
        // did not wire device_memset.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        int rc;
        if (is_pure_output && api->device_memset != nullptr) {
            rc = api->device_memset(dev_ptr, 0, size);
        } else {
            rc = api->copy_to_device(dev_ptr, host_ptr, size);
        }
        if (rc != 0) {
            LOG_ERROR("Failed to stage tensor %d to device", i);
            if (release_kind == TensorReleaseKind::Free) {
                api->device_free(dev_ptr);
            }
            return false;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (child_memory tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        bool host_mapped = false;
        if (map_for_host_orch) {
            if (api->register_device_memory_to_host == nullptr || api->unregister_device_memory_from_host == nullptr) {
                LOG_ERROR("Host orchestration requires device-memory host mapping support");
                if (release_kind == TensorReleaseKind::Free) api->device_free(dev_ptr);
                return false;
            }
            void *host_va = api->register_device_memory_to_host(dev_ptr, size);
            if (host_va != dev_ptr) {
                LOG_ERROR(
                    "Host orchestration requires identity SVM mapping: device=%p host=%p", dev_ptr, host_va
                );
                if (host_va != nullptr) api->unregister_device_memory_from_host(dev_ptr);
                if (release_kind == TensorReleaseKind::Free) api->device_free(dev_ptr);
                return false;
            }
            host_mapped = true;
        }
        runtime->tensor_leases_.push_back({host_ptr, dev_ptr, size, needs_copy_back, release_kind, host_mapped});
        LOG_INFO_V0("  Tensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        out->add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        out->add_scalar(orch_args->scalar(i));
    }
    int64_t t_args_end = _now_ms();
    LOG_INFO_V0("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    return true;
}

// per-(cid,config): reserve and acquire the static device pools. GM heap, PTO2
// shared memory, and the prebuilt runtime arena all live in one backing
// allocation; setup_static_arena reserves the three regions and commits in one
// shot. The runtime-arena size is recovered by replaying the (pure, cheap)
// reserve sequence on a throwaway host arena. Idempotent across runs — the
// pools are owned by DeviceRunner and freed in DeviceRunner::finalize().
static bool ensure_static_arenas(
    Runtime *runtime, const HostApi *api, const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes,
    StaticArenaPtrs *out
) {
    DeviceArena sizing_arena;  // discarded; only its computed arena_size is read
    PTO2RuntimeArenaLayout layout =
        runtime_reserve_layout(sizing_arena, sizing.task_window_size, sizing.heap_size, sizing.dep_pool_capacity);

    int64_t t_setup_start = _now_ms();
    if (api->setup_static_arena(sizes.heap_size, sizes.sm_size, layout.offsets.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return false;
    }
    int64_t t_setup_end = _now_ms();

    int64_t t_heap_start = _now_ms();
    out->gm_heap = api->acquire_pooled_gm_heap();
    int64_t t_heap_end = _now_ms();
    if (out->gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return false;
    }

    int64_t t_sm_start = _now_ms();
    out->gm_sm = api->acquire_pooled_gm_sm();
    int64_t t_sm_end = _now_ms();
    if (out->gm_sm == nullptr) {
        LOG_ERROR("Failed to acquire pooled PTO2 shared memory");
        return false;
    }
    runtime->set_gm_sm_ptr(out->gm_sm);

    out->runtime_arena_dev = api->acquire_pooled_runtime_arena();
    if (out->runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return false;
    }

    LOG_INFO_V0("TIMING: static_arena_setup = %" PRId64 "ms", t_setup_end - t_setup_start);
    LOG_INFO_V0("TIMING: gm_heap_acquire = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO_V0("TIMING: shared_mem_acquire = %" PRId64 "ms", t_sm_end - t_sm_start);
    return true;
}

// per-(cid,config): build the prebuilt runtime-arena image on host. Pure host
// work — touches no device memory, only `host_arena` (owned by the caller so
// the image outlives this call until the upload) and the device *addresses* in
// `ptrs` (stored, never dereferenced).
//
// We pre-compute every byte the AICPU's runtime arena would otherwise have to
// write at boot: layout offsets, sub-structure init data, and pointers back to
// the SM / GM heap. AICPU boot then becomes attach + wire (cheap pointer fixup)
// + sm_handle->init (SM reset) + a handful of device-only field fixups.
//
// The layout is stashed inside the image (rt->prebuilt_layout) so the AICPU can
// recover every arena-internal offset after the rtMemcpy. Returns the layout
// via `out_layout`; the runtime-arena device base travels separately on the
// host Runtime (bind_launch_state), since the AICPU needs that pointer *before*
// it can dereference the image.
static bool build_runtime_image(
    const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes, const StaticArenaPtrs &ptrs,
    DeviceArena *host_arena, PTO2RuntimeArenaLayout *out_layout
) {
    PTO2RuntimeArenaLayout layout =
        runtime_reserve_layout(*host_arena, sizing.task_window_size, sizing.heap_size, sizing.dep_pool_capacity);
    if (host_arena->commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return false;
    }

    PTO2Runtime *rt = runtime_init_data_from_layout(
        *host_arena, layout, PTO2_MODE_EXECUTE, ptrs.gm_sm, sizes.sm_size, ptrs.gm_heap, sizing.heap_size
    );
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return false;
    }
    runtime_wire_arena_pointers(*host_arena, layout, rt);
    rt->prebuilt_layout = layout;

    *out_layout = layout;
    return true;
}

// per-run: publish the launch state. Copy the staged args onto the runtime,
// rtMemcpy the host image into the pooled runtime-arena region, and record the
// device base + runtime offset the AICPU reads before dereferencing the image.
static bool bind_launch_state(
    Runtime *runtime, const HostApi *api, const StaticArenaPtrs &ptrs, const DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, const ChipStorageTaskArgs &device_args
) {
    runtime->set_orch_args(device_args);

    int rc_upload = api->copy_to_device(ptrs.runtime_arena_dev, host_arena.base(), layout.offsets.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return false;
    }
    runtime->set_prebuilt_arena(ptrs.runtime_arena_dev, layout.offsets.off_runtime);
    return true;
}

static int bind_cached_runtime_image(
    Runtime *runtime, const HostApi *api, const PrebuiltRuntimeArenaCacheProbe &probe,
    const ChipStorageTaskArgs &device_args
) {
    if (api->lookup_prebuilt_runtime_arena_cache == nullptr) {
        return 1;
    }

    void *gm_heap = nullptr;
    void *sm_ptr = nullptr;
    void *runtime_arena_dev = nullptr;
    size_t runtime_off = 0;
    const void *cached_image = nullptr;
    size_t cached_image_size = 0;
    bool cache_hit = api->lookup_prebuilt_runtime_arena_cache(
        probe.hash, probe.serialized_key.data(), probe.serialized_key.size(), &gm_heap, &sm_ptr, &runtime_arena_dev,
        &runtime_off, &cached_image, &cached_image_size
    );
    if (!cache_hit) {
        return 1;
    }

    runtime->set_orch_args(device_args);
    (void)cached_image;
    (void)cached_image_size;
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->set_prebuilt_arena(runtime_arena_dev, runtime_off);
    return 0;
}

static void store_prebuilt_runtime_image(
    Runtime *runtime, const HostApi *api, const PrebuiltRuntimeArenaCacheProbe &probe, const StaticArenaPtrs &ptrs,
    const PTO2RuntimeArenaLayout &layout, const DeviceArena &host_arena
) {
    if (api->mark_prebuilt_runtime_arena_cached == nullptr) {
        return;
    }
    api->mark_prebuilt_runtime_arena_cached(
        probe.hash, probe.serialized_key.data(), probe.serialized_key.size(), ptrs.gm_heap, ptrs.gm_sm,
        ptrs.runtime_arena_dev, layout.offsets.off_runtime, host_arena.base(), layout.offsets.arena_size
    );
}

struct HostGraphRelocation {
    uint64_t host_sm;
    uint64_t device_sm;
    uint64_t sm_size;
    uint64_t host_arena;
    uint64_t device_arena;
    uint64_t arena_size;
};

static uint64_t relocate_host_graph_pointer(const HostGraphRelocation &reloc, uint64_t value, bool *ok) {
    if (value == 0 || value == reinterpret_cast<uint64_t>(pto2_fanout_closed_sentinel())) return value;
    if (value >= reloc.host_sm && value < reloc.host_sm + reloc.sm_size) {
        return reloc.device_sm + (value - reloc.host_sm);
    }
    if (value >= reloc.host_arena && value < reloc.host_arena + reloc.arena_size) {
        return reloc.device_arena + (value - reloc.host_arena);
    }
    LOG_ERROR("Host graph contains a pointer outside the SM/runtime arena: 0x%" PRIx64, value);
    *ok = false;
    return 0;
}

template <typename T>
static T *relocate_host_graph_pointer(const HostGraphRelocation &reloc, T *value, bool *ok) {
    return reinterpret_cast<T *>(static_cast<uintptr_t>(
        relocate_host_graph_pointer(reloc, reinterpret_cast<uint64_t>(value), ok)
    ));
}

class HostGraphAsyncJob;
static void record_host_graph_boundary(PTO2Runtime *rt);

struct HostGraphCaptureContext {
    PTO2Runtime *source{nullptr};
    HostGraphAsyncJob *async_job{nullptr};
};

static thread_local HostGraphCaptureContext *g_host_graph_capture_context = nullptr;

class HostGraphAsyncJob {
public:
    ~HostGraphAsyncJob() { (void)join(); }

    bool prepare_and_start(
        Runtime *runtime, const HostApi *api, const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes,
        const StaticArenaPtrs &ptrs, const ChipStorageTaskArgs &device_args, void *host_orch_func_ptr
    ) {
        runtime_ = runtime;
        api_ = api;
        sizing_ = sizing;
        sizes_ = sizes;
        ptrs_ = ptrs;
        device_args_ = device_args;
        entry_points_ = static_cast<HostOrchEntryPoints *>(host_orch_func_ptr);
#if SIMPLER_HOST_STRACE
        strace_inv_ = simpler::strace::StraceScope::current_inv();
        strace_hid_ = simpler::strace::StraceScope::current_hid();
#endif
        if (runtime_ == nullptr || api_ == nullptr || entry_points_ == nullptr || entry_points_->entry == nullptr ||
            entry_points_->bind == nullptr || api_->capture_thread_context == nullptr ||
            api_->bind_thread_context == nullptr || api_->unbind_thread_context == nullptr) {
            LOG_ERROR("HostGraph async job has incomplete inputs");
            return false;
        }
        thread_context_ = api_->capture_thread_context();
        if (thread_context_ == nullptr) {
            LOG_ERROR("HostGraph async job could not capture the platform thread context");
            return false;
        }

        source_layout_ = runtime_reserve_layout(
            source_arena_, sizing_.task_window_size, sizing_.heap_size, sizing_.dep_pool_capacity
        );
        target_layout_ = runtime_reserve_layout(
            target_arena_, sizing_.task_window_size, sizing_.heap_size, sizing_.dep_pool_capacity
        );
        if (source_arena_.commit(DeviceArena::kDefaultBaseAlign) == nullptr ||
            target_arena_.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            LOG_ERROR("Failed to commit HostGraph async arenas");
            return false;
        }

        source_sm_.resize(static_cast<size_t>(sizes_.sm_size));
        target_sm_.resize(static_cast<size_t>(sizes_.sm_size));
        source_ = runtime_init_data_from_layout(
            source_arena_, source_layout_, PTO2_MODE_EXECUTE, source_sm_.data(), sizes_.sm_size, ptrs_.gm_heap,
            sizing_.heap_size
        );
        target_ = runtime_init_data_from_layout(
            target_arena_, target_layout_, PTO2_MODE_EXECUTE, target_sm_.data(), sizes_.sm_size, ptrs_.gm_heap,
            sizing_.heap_size
        );
        if (source_ == nullptr || target_ == nullptr) {
            LOG_ERROR("Failed to initialize HostGraph async runtimes");
            return false;
        }
        runtime_wire_arena_pointers(source_arena_, source_layout_, source_);
        runtime_wire_arena_pointers(target_arena_, target_layout_, target_);
        if (!source_->sm_handle->init(
                source_sm_.data(), sizes_.sm_size, sizing_.task_window_size, sizing_.heap_size
            ) ||
            !target_->sm_handle->init(
                target_sm_.data(), sizes_.sm_size, sizing_.task_window_size, sizing_.heap_size
            )) {
            LOG_ERROR("Failed to initialize HostGraph async shared memory");
            return false;
        }
        runtime_finalize_after_wire(source_, 24, 48);
        runtime_finalize_after_wire(target_, 24, 48);
        source_->graph_cache_enabled = runtime_->graph_cache_enabled();
        source_->active_callable_hash = runtime_->active_callable_hash();
        source_->graph_cache_stats = PTO2GraphCacheStats{};
        target_->prebuilt_layout = target_layout_;

        // sm_header is outside the runtime arena, so arena pointer wiring does
        // not rewrite it on AICPU. Upload the device address, then restore the
        // host mirror address so the range importer can keep using target_.
        PTO2SharedMemoryHeader *target_host_header = target_->orchestrator.sm_header;
        target_->orchestrator.sm_header = static_cast<PTO2SharedMemoryHeader *>(ptrs_.gm_sm);
        target_->scheduler.sm_header = static_cast<PTO2SharedMemoryHeader *>(ptrs_.gm_sm);
        const int arena_upload_rc = api_->copy_to_device(
                ptrs_.runtime_arena_dev, target_arena_.base(), target_layout_.offsets.arena_size
            );
        target_->orchestrator.sm_header = target_host_header;
        target_->scheduler.sm_header = target_host_header;
        if (arena_upload_rc != 0 || api_->copy_to_device(ptrs_.gm_sm, target_sm_.data(), sizes_.sm_size) != 0) {
            LOG_ERROR("Failed to upload the empty HostGraph pipeline image");
            return false;
        }

        runtime_->set_orch_args(device_args_);
        runtime_->set_prebuilt_arena(ptrs_.runtime_arena_dev, target_layout_.offsets.off_runtime);
        runtime_->set_host_orch_result(-1);
        try {
            worker_ = std::thread([this]() {
#if SIMPLER_HOST_STRACE
                STRACE_SET_CONTEXT(strace_inv_, strace_hid_);
#endif
                if (api_->bind_thread_context(thread_context_) != 0) {
                    LOG_ERROR("HostGraph async worker failed to bind the platform thread context");
                    result_ = -1;
                    worker_start_state_.store(-1, std::memory_order_release);
                    return;
                }
                worker_start_state_.store(1, std::memory_order_release);
                {
                    std::unique_lock<std::mutex> lock(run_gate_mutex_);
                    run_gate_cv_.wait(lock, [this]() {
                        return run_gate_state_.load(std::memory_order_acquire) != 0;
                    });
                }
                if (run_gate_state_.load(std::memory_order_acquire) < 0) {
                    api_->unbind_thread_context();
                    return;
                }
                try {
                    run();
                } catch (const std::exception &e) {
                    LOG_ERROR("HostGraph async worker threw an exception: %s", e.what());
                    publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
                    result_ = -1;
                } catch (...) {
                    LOG_ERROR("HostGraph async worker threw an unknown exception");
                    publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
                    result_ = -1;
                }
                api_->unbind_thread_context();
            });
        } catch (const std::exception &e) {
            LOG_ERROR("Failed to start HostGraph async worker: %s", e.what());
            return false;
        }
        while (worker_start_state_.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        if (worker_start_state_.load(std::memory_order_acquire) < 0) {
            worker_.join();
            return false;
        }
        return true;
    }

    bool release_for_run() {
        int32_t expected = 0;
        if (!run_gate_state_.compare_exchange_strong(
                expected, 1, std::memory_order_release, std::memory_order_acquire
            )) {
            return expected == 1;
        }
        run_gate_cv_.notify_one();
        LOG_INFO_V0("HostGraph async worker released after runner profiling setup");
        return true;
    }

    bool publish_boundary(bool final_publish) {
        if (source_ == nullptr || target_ == nullptr || source_->orchestrator.sm_header == nullptr) return false;
        const int32_t layer = publish_epoch_;
        const int64_t boundary_start_ns = _steady_now_ns();
        const int32_t task_end =
            source_->orchestrator.sm_header->fc.task_count.load(std::memory_order_acquire);
        const int32_t task_count = task_end - task_begin_;
        const int64_t build_end_ns = boundary_start_ns;
        const int64_t build_ns = task_count > 0 && layer_build_start_ns_ > 0 ? build_end_ns - layer_build_start_ns_ : 0;
        const uint64_t recorded = source_->graph_cache_stats.recorded;
        const uint64_t replayed = source_->graph_cache_stats.replayed;
        const bool recorded_this_layer = recorded > last_cache_recorded_;
        const bool replayed_this_layer = replayed > last_cache_replayed_;
        const char *cache_mode =
            recorded_this_layer ? (replayed_this_layer ? "mixed" : "record")
                                : (replayed_this_layer ? "replay" : "none");
        last_cache_recorded_ = recorded;
        last_cache_replayed_ = replayed;
        char trace_attrs[160];
        std::snprintf(
            trace_attrs, sizeof(trace_attrs), "layer=%d tasks=%d final=%d cache=%s", layer, task_count,
            final_publish ? 1 : 0, cache_mode
        );
        if (build_ns > 0) {
            STRACE_HOST_SPAN_AT(
                "simpler_run.host_orch.layer_build", layer_build_start_ns_, build_ns, 1, trace_attrs
            );
        }

        STRACE_A("simpler_run.host_orch.boundary", trace_attrs);
        boundary_h2d_calls_ = 0;
        boundary_h2d_bytes_ = 0;
        boundary_d2h_reads_ = 0;

        HostGraphHeader graph_header;
        const int32_t dep_begin = target_->orchestrator.dep_pool.top;
        const int64_t materialize_start_ns = _steady_now_ns();
        {
            STRACE_A("simpler_run.host_orch.boundary.materialize", trace_attrs);
            if (!materialize_host_graph_range(source_, task_begin_, task_end, target_, &graph_header)) {
                LOG_ERROR("Failed to materialize HostGraph range [%d,%d)", task_begin_, task_end);
                return false;
            }
        }
        const int64_t materialize_end_ns = _steady_now_ns();
        const int32_t dep_end = target_->orchestrator.dep_pool.top;

        // Stage the inactive graph buffer while Device S executes the current
        // range. The epoch remains unchanged, so Device S cannot observe the
        // staged range until commit_graph_publication() writes it last.
        const int64_t stage_upload_start_ns = materialize_end_ns;
        {
            STRACE_A("simpler_run.host_orch.boundary.stage_upload", trace_attrs);
            if (!stage_graph_bytes(graph_header, dep_begin, dep_end)) return false;
        }
        const int64_t stage_upload_end_ns = _steady_now_ns();

        // A full previous-graph barrier replaces cross-range producer edges.
        // Only the publication metadata remains after this barrier.
        const int64_t wait_start_ns = stage_upload_end_ns;
        {
            STRACE_A("simpler_run.host_orch.boundary.wait", trace_attrs);
            if (!wait_for_completion(publish_epoch_)) return false;
        }
        const int64_t wait_end_ns = _steady_now_ns();

        const int32_t next_epoch = publish_epoch_ + 1;
        const int64_t commit_start_ns = wait_end_ns;
        {
            STRACE_A("simpler_run.host_orch.boundary.commit", trace_attrs);
            if (!commit_graph_publication(graph_header, next_epoch, final_publish)) return false;
        }
        const int64_t commit_end_ns = _steady_now_ns();
        const int64_t release_wait_start_ns = commit_end_ns;
        if (!final_publish) {
            STRACE_A("simpler_run.host_orch.boundary.release_wait", trace_attrs);
            if (!wait_for_release(next_epoch)) return false;
        }
        const int64_t release_wait_end_ns = _steady_now_ns();
        const int64_t active_ns = build_ns + (materialize_end_ns - materialize_start_ns) +
                                  (stage_upload_end_ns - stage_upload_start_ns) +
                                  (commit_end_ns - commit_start_ns);
        publish_epoch_ = next_epoch;
        LOG_INFO_V0(
            "HostGraph async publish epoch=%d buffer=%d tasks=[%d,%d) edges=%d final=%d", publish_epoch_,
            buffer_id_, task_begin_, task_end, graph_header.edge_count, final_publish ? 1 : 0
        );
        LOG_INFO_V0(
            "[HostGraphProfile] layer=%d cache=%s tasks=%d edges=%d final=%d active_us=%.3f build_us=%.3f "
            "materialize_us=%.3f stage_upload_us=%.3f wait_us=%.3f commit_us=%.3f "
            "release_wait_us=%.3f h2d_calls=%llu h2d_bytes=%llu d2h_reads=%llu",
            layer, cache_mode, task_count, graph_header.edge_count, final_publish ? 1 : 0, active_ns / 1000.0,
            build_ns / 1000.0,
            (materialize_end_ns - materialize_start_ns) / 1000.0,
            (stage_upload_end_ns - stage_upload_start_ns) / 1000.0,
            (wait_end_ns - wait_start_ns) / 1000.0, (commit_end_ns - commit_start_ns) / 1000.0,
            (release_wait_end_ns - release_wait_start_ns) / 1000.0,
            static_cast<unsigned long long>(boundary_h2d_calls_),
            static_cast<unsigned long long>(boundary_h2d_bytes_),
            static_cast<unsigned long long>(boundary_d2h_reads_)
        );

        if (!final_publish) {
            const int32_t next_buffer = (buffer_id_ + 1) % PTO2_REPLAY_GRAPH_BUFFER_COUNT;
            source_->orchestrator.task_allocator.begin_buffer(next_buffer);
            source_->graph_pipeline.active_buffer = next_buffer;
            source_->graph_pipeline.buffers[next_buffer].task_begin = task_end;
            source_->graph_pipeline.buffers[next_buffer].graph_epoch = static_cast<uint64_t>(publish_epoch_);
            target_->orchestrator.task_allocator.begin_buffer(next_buffer);
            buffer_id_ = next_buffer;
            task_begin_ = task_end;
            layer_build_start_ns_ = _steady_now_ns();
        }
        return true;
    }

    int join() {
        int32_t expected = 0;
        if (run_gate_state_.compare_exchange_strong(
                expected, -1, std::memory_order_release, std::memory_order_acquire
            )) {
            run_gate_cv_.notify_one();
        }
        if (worker_.joinable()) worker_.join();
        return result_;
    }

private:
    bool copy_sm(size_t offset, const void *src, size_t size) const {
        if (offset + size > sizes_.sm_size) {
            LOG_ERROR("HostGraph SM copy is out of bounds (offset=%zu size=%zu total=%llu)", offset, size,
                      static_cast<unsigned long long>(sizes_.sm_size));
            return false;
        }
        void *dst = static_cast<void *>(static_cast<char *>(ptrs_.gm_sm) + offset);
        const int rc = api_->copy_to_device(dst, src, size);
        boundary_h2d_calls_++;
        boundary_h2d_bytes_ += size;
        if (rc != 0) {
            LOG_ERROR("HostGraph SM H2D failed (offset=%zu size=%zu rc=%d)", offset, size, rc);
            return false;
        }
        return true;
    }

    bool copy_arena(const void *host_ptr, const void *src, size_t size) const {
        const char *base = static_cast<const char *>(target_arena_.base());
        const char *ptr = static_cast<const char *>(host_ptr);
        if (ptr < base || ptr + size > base + target_layout_.offsets.arena_size) {
            LOG_ERROR("HostGraph arena copy is out of bounds (size=%zu)", size);
            return false;
        }
        const size_t offset = static_cast<size_t>(ptr - base);
        void *dst = static_cast<void *>(
            static_cast<char *>(ptrs_.runtime_arena_dev) + offset
        );
        const int rc = api_->copy_to_device(dst, src, size);
        boundary_h2d_calls_++;
        boundary_h2d_bytes_ += size;
        if (rc != 0) {
            LOG_ERROR("HostGraph arena H2D failed (offset=%zu size=%zu rc=%d)", offset, size, rc);
            return false;
        }
        return true;
    }

    int32_t read_device_i32(size_t offset, int32_t fallback) const {
        if (api_->copy_from_device == nullptr) return fallback;
        int32_t value = fallback;
        const void *src = static_cast<const void *>(static_cast<const char *>(ptrs_.gm_sm) + offset);
        boundary_d2h_reads_++;
        return api_->copy_from_device(&value, src, sizeof(value)) == 0 ? value : fallback;
    }

    bool wait_for_device_epoch(size_t offset, int32_t epoch, const char *kind) const {
        if (epoch <= 0) return true;
        const int64_t deadline = _now_ms() + 120000;
        while (read_device_i32(offset, 0) < epoch) {
            const int32_t orch_error =
                read_device_i32(offsetof(PTO2SharedMemoryHeader, orch_error_code), PTO2_ERROR_NONE);
            const int32_t sched_error =
                read_device_i32(offsetof(PTO2SharedMemoryHeader, sched_error_code), PTO2_ERROR_NONE);
            if (orch_error != PTO2_ERROR_NONE || sched_error != PTO2_ERROR_NONE) {
                LOG_ERROR(
                    "HostGraph %s wait failed at epoch=%d (orch=%d sched=%d)", kind, epoch, orch_error,
                    sched_error
                );
                return false;
            }
            if (_now_ms() >= deadline) {
                LOG_ERROR("HostGraph %s wait timed out at epoch=%d", kind, epoch);
                return false;
            }
            usleep(50);
        }
        return true;
    }

    bool wait_for_completion(int32_t epoch) const {
        return wait_for_device_epoch(
            offsetof(PTO2SharedMemoryHeader, device_graph_complete_epoch), epoch, "completion"
        );
    }

    bool wait_for_release(int32_t epoch) const {
        return wait_for_device_epoch(
            offsetof(PTO2SharedMemoryHeader, device_graph_release_epoch), epoch, "release"
        );
    }

    bool stage_graph_bytes(const HostGraphHeader &graph, int32_t dep_begin, int32_t dep_end) {
        HostGraphRelocation reloc{
            reinterpret_cast<uint64_t>(target_sm_.data()), reinterpret_cast<uint64_t>(ptrs_.gm_sm), sizes_.sm_size,
            reinterpret_cast<uint64_t>(target_arena_.base()),
            reinterpret_cast<uint64_t>(ptrs_.runtime_arena_dev), target_layout_.offsets.arena_size,
        };
        bool ok = true;
        PTO2DepListPool &dep_pool = target_->orchestrator.dep_pool;
        if (dep_end > dep_begin) {
            const size_t dep_count = static_cast<size_t>(dep_end - dep_begin);
            std::vector<uint8_t> dep_bytes(dep_count * sizeof(PTO2DepListEntry));
            for (int32_t i = dep_begin; i < dep_end; i++) {
                PTO2DepListEntry tmp = dep_pool.base[i];
                tmp.slot_state = relocate_host_graph_pointer(reloc, tmp.slot_state, &ok);
                tmp.next = relocate_host_graph_pointer(reloc, tmp.next, &ok);
                if (!ok) return false;
                std::memcpy(
                    dep_bytes.data() + static_cast<size_t>(i - dep_begin) * sizeof(tmp), &tmp, sizeof(tmp)
                );
            }
            if (!copy_arena(&dep_pool.base[dep_begin], dep_bytes.data(), dep_bytes.size())) return false;
        }

        PTO2SharedMemoryHeader *header = target_->orchestrator.sm_header;
        struct TaskUploadRun {
            int32_t first_slot{-1};
            int32_t count{0};
            std::vector<uint8_t> slot_bytes;
        };
        std::vector<TaskUploadRun> task_runs;
        for (int32_t task_id = graph.first_task_id; task_id < graph.first_task_id + graph.task_count; task_id++) {
            const int32_t slot_id = header->get_slot_by_task_id(task_id);
            if (slot_id < 0 || slot_id >= static_cast<int32_t>(header->task_window_size)) {
                LOG_ERROR("HostGraph task slot is invalid (task=%d slot=%d window=%llu)", task_id, slot_id,
                          static_cast<unsigned long long>(header->task_window_size));
                return false;
            }
            PTO2TaskSlotState &slot = header->get_slot_state_by_slot(slot_id);
            PTO2TaskSlotState slot_copy;
            std::memcpy(&slot_copy, &slot, sizeof(slot_copy));
            slot_copy.task = relocate_host_graph_pointer(reloc, slot_copy.task, &ok);
            slot_copy.payload = relocate_host_graph_pointer(reloc, slot_copy.payload, &ok);
            PTO2DepListEntry *fanout = slot.fanout_head.load(std::memory_order_relaxed);
            slot_copy.fanout_head.store(
                relocate_host_graph_pointer(reloc, fanout, &ok), std::memory_order_relaxed
            );
            if (!ok) return false;

            if (task_runs.empty() || slot_id != task_runs.back().first_slot + task_runs.back().count) {
                task_runs.push_back(TaskUploadRun{});
                task_runs.back().first_slot = slot_id;
            }
            TaskUploadRun &run = task_runs.back();
            const size_t slot_offset = static_cast<size_t>(run.count) * sizeof(slot_copy);
            run.slot_bytes.resize(slot_offset + sizeof(slot_copy));
            std::memcpy(run.slot_bytes.data() + slot_offset, &slot_copy, sizeof(slot_copy));
            run.count++;
        }

        const char *sm_host_base = reinterpret_cast<const char *>(target_sm_.data());
        for (const TaskUploadRun &run : task_runs) {
            const size_t task_bytes = static_cast<size_t>(run.count) * sizeof(PTO2TaskDescriptor);
            const size_t payload_bytes = static_cast<size_t>(run.count) * sizeof(PTO2TaskPayload);
            const size_t slot_bytes = static_cast<size_t>(run.count) * sizeof(PTO2TaskSlotState);
            PTO2TaskDescriptor *tasks = &header->task_descriptors[run.first_slot];
            PTO2TaskPayload *payloads = &header->task_payloads[run.first_slot];
            PTO2TaskSlotState *slots = &header->slot_states[run.first_slot];
            if (!copy_sm(static_cast<size_t>(reinterpret_cast<const char *>(tasks) - sm_host_base), tasks, task_bytes) ||
                !copy_sm(
                    static_cast<size_t>(reinterpret_cast<const char *>(payloads) - sm_host_base), payloads,
                    payload_bytes
                ) ||
                !copy_sm(
                    static_cast<size_t>(reinterpret_cast<const char *>(slots) - sm_host_base),
                    run.slot_bytes.data(), slot_bytes
                )) {
                return false;
            }
        }

        const int32_t task_count = graph.task_count;
        if (task_count > 0) {
            if (task_count > static_cast<int32_t>(header->task_window_size)) {
                LOG_ERROR(
                    "HostGraph task range exceeds the slot map (tasks=%d window=%llu)", task_count,
                    static_cast<unsigned long long>(header->task_window_size)
                );
                return false;
            }
            const int32_t map_begin = graph.first_task_id & header->task_window_mask;
            const int32_t first_count =
                task_count < static_cast<int32_t>(header->task_window_size) - map_begin
                    ? task_count
                    : static_cast<int32_t>(header->task_window_size) - map_begin;
            if (!copy_sm(
                    static_cast<size_t>(reinterpret_cast<const char *>(&header->task_slot_map[map_begin]) -
                                        sm_host_base),
                    &header->task_slot_map[map_begin], static_cast<size_t>(first_count) * sizeof(int32_t)
                )) {
                return false;
            }
            const int32_t second_count = task_count - first_count;
            if (second_count > 0 &&
                !copy_sm(
                    static_cast<size_t>(reinterpret_cast<const char *>(header->task_slot_map) - sm_host_base),
                    header->task_slot_map, static_cast<size_t>(second_count) * sizeof(int32_t)
                )) {
                return false;
            }
        }

        const int64_t inline_completed = target_->orchestrator.inline_completed_tasks;
        if (!copy_arena(
                &target_->orchestrator.inline_completed_tasks, &inline_completed, sizeof(inline_completed)
            )) {
            return false;
        }

        const uint64_t output_values[2] = {graph.graph_output_ptr, graph.graph_output_size};
        static_assert(
            offsetof(PTO2SharedMemoryHeader, graph_output_size) ==
            offsetof(PTO2SharedMemoryHeader, graph_output_ptr) + sizeof(std::atomic<uint64_t>)
        );
        if (!copy_sm(
                offsetof(PTO2SharedMemoryHeader, graph_output_ptr), output_values, sizeof(output_values)
            )) {
            LOG_ERROR(
                "Failed to stage HostGraph range [%d,%d)", graph.first_task_id,
                graph.first_task_id + graph.task_count
            );
            return false;
        }
        return true;
    }

    bool commit_graph_publication(const HostGraphHeader &graph, int32_t epoch, bool final_publish) const {
        const int32_t task_begin = graph.first_task_id;
        const int32_t task_end = graph.first_task_id + graph.task_count;
        const int32_t final_value = final_publish ? 1 : 0;
        const int32_t done_value = final_publish ? 1 : 0;
        const int32_t range_values[3] = {task_begin, task_end, final_value};
        static_assert(
            offsetof(PTO2SharedMemoryHeader, host_graph_task_end) ==
            offsetof(PTO2SharedMemoryHeader, host_graph_task_begin) + sizeof(std::atomic<int32_t>)
        );
        static_assert(
            offsetof(PTO2SharedMemoryHeader, host_graph_final) ==
            offsetof(PTO2SharedMemoryHeader, host_graph_task_end) + sizeof(std::atomic<int32_t>)
        );
        if (!copy_sm(
                offsetof(PTO2SharedMemoryHeader, host_graph_task_begin), range_values, sizeof(range_values)
            ) ||
            !copy_sm(
                offsetof(PTO2SharedMemoryHeader, fc) + offsetof(PTO2FlowControl, task_count),
                &task_end, sizeof(task_end)
            ) ||
            (final_publish &&
             !copy_sm(offsetof(PTO2SharedMemoryHeader, orchestrator_done), &done_value, sizeof(done_value))) ||
            !copy_sm(offsetof(PTO2SharedMemoryHeader, host_graph_publish_epoch), &epoch, sizeof(epoch))) {
            LOG_ERROR("Failed to commit HostGraph epoch=%d", epoch);
            return false;
        }
        return true;
    }

    void publish_failure(int32_t error_code) const {
        (void)copy_sm(offsetof(PTO2SharedMemoryHeader, orch_error_code), &error_code, sizeof(error_code));
    }

    void run() {
        PTO2RuntimeOps host_ops = *source_->ops;
        host_ops.graph_boundary = record_host_graph_boundary;
        source_->ops = &host_ops;
        HostGraphCaptureContext capture_context;
        capture_context.source = source_;
        capture_context.async_job = this;
        g_host_graph_capture_context = &capture_context;
        framework_bind_runtime(source_);
        entry_points_->bind(source_);

        L2TaskArgs host_args;
        host_args.create_from_chip_args(device_args_);
        rt_scope_begin(source_);
        layer_build_start_ns_ = _steady_now_ns();
        entry_points_->entry(host_args);
        rt_scope_end(source_);
        if (source_->graph_cache_enabled) {
            LOG_INFO_V0(
                "[HostGraphCache] stats recorded=%llu missed=%llu replayed=%llu overflow=%llu",
                static_cast<unsigned long long>(source_->graph_cache_stats.recorded),
                static_cast<unsigned long long>(source_->graph_cache_stats.missed),
                static_cast<unsigned long long>(source_->graph_cache_stats.replayed),
                static_cast<unsigned long long>(source_->graph_cache_stats.overflow)
            );
        }
        if (!source_->orchestrator.fatal && !publish_boundary(true)) {
            source_->orchestrator.report_fatal(
                PTO2_ERROR_EXPLICIT_ORCH_FATAL, __FUNCTION__, "final HostGraph publication failed"
            );
        }

        entry_points_->bind(nullptr);
        framework_bind_runtime(nullptr);
        g_host_graph_capture_context = nullptr;
        if (source_->orchestrator.fatal) {
            publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
            result_ = -1;
            return;
        }
        result_ = 0;
    }

    Runtime *runtime_{nullptr};
    const HostApi *api_{nullptr};
    ArenaSizingConfig sizing_{};
    ArenaStaticSizes sizes_{};
    StaticArenaPtrs ptrs_{};
    ChipStorageTaskArgs device_args_{};
    HostOrchEntryPoints *entry_points_{nullptr};
    void *thread_context_{nullptr};
    DeviceArena source_arena_;
    DeviceArena target_arena_;
    PTO2RuntimeArenaLayout source_layout_{};
    PTO2RuntimeArenaLayout target_layout_{};
    std::vector<uint8_t> source_sm_;
    std::vector<uint8_t> target_sm_;
    PTO2Runtime *source_{nullptr};
    PTO2Runtime *target_{nullptr};
    std::thread worker_;
    std::atomic<int32_t> worker_start_state_{0};
    std::atomic<int32_t> run_gate_state_{0};
    std::mutex run_gate_mutex_;
    std::condition_variable run_gate_cv_;
    int32_t buffer_id_{0};
    int32_t task_begin_{0};
    int32_t publish_epoch_{0};
    int64_t layer_build_start_ns_{0};
    mutable uint64_t boundary_h2d_calls_{0};
    mutable uint64_t boundary_h2d_bytes_{0};
    mutable uint64_t boundary_d2h_reads_{0};
    uint64_t last_cache_recorded_{0};
    uint64_t last_cache_replayed_{0};
#if SIMPLER_HOST_STRACE
    unsigned strace_inv_{0};
    uint64_t strace_hid_{0};
#endif
    int result_{-1};
};

static void record_host_graph_boundary(PTO2Runtime *rt) {
    HostGraphCaptureContext *ctx = g_host_graph_capture_context;
    if (rt == nullptr || ctx == nullptr || ctx->source != rt || rt->orchestrator.sm_header == nullptr) {
        if (rt != nullptr) {
            rt->orchestrator.report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "HostGraph boundary callback has no active capture context"
            );
        }
        return;
    }
    if (ctx->async_job == nullptr) {
        rt->orchestrator.report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "HostGraph boundary callback has no async job"
        );
        return;
    }
    if (!ctx->async_job->publish_boundary(false)) {
        rt->orchestrator.report_fatal(
            PTO2_ERROR_EXPLICIT_ORCH_FATAL, __FUNCTION__, "HostGraph boundary publication failed"
        );
    }
}

static bool start_async_host_graph_pipeline(
    Runtime *runtime, const HostApi *api, const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes,
    const StaticArenaPtrs &ptrs, const ChipStorageTaskArgs &device_args, void *host_orch_func_ptr
) {
    auto *job = new (std::nothrow) HostGraphAsyncJob{};
    if (job == nullptr) {
        LOG_ERROR("Failed to allocate HostGraph async job");
        return false;
    }
    if (!job->prepare_and_start(runtime, api, sizing, sizes, ptrs, device_args, host_orch_func_ptr)) {
        delete job;
        return false;
    }
    runtime->set_host_orch_job(job);
    return true;
}

static int join_async_host_graph_pipeline(Runtime *runtime) {
    auto *job = static_cast<HostGraphAsyncJob *>(runtime->take_host_orch_job());
    if (job == nullptr) return 0;
    const int rc = job->join();
    delete job;
    return rc;
}

extern "C" int release_async_host_graph_pipeline(Runtime *runtime) {
    if (runtime == nullptr) return -1;
    auto *job = static_cast<HostGraphAsyncJob *>(runtime->take_host_orch_job());
    if (job == nullptr) return 0;
    runtime->set_host_orch_job(job);
    return job->release_for_run() ? 0 : -1;
}

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, PTO2 shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/run invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * Orchestrates the three lifecycles behind the bind: per-config arena sizing
 * (resolve_arena_sizing) + static pools (ensure_static_arenas) + host image
 * (build_runtime_image), and per-run args (stage_device_args) + launch publish
 * (bind_launch_state).
 *
 * @param runtime    Pointer to pre-constructed Runtime
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }
    const bool host_orch_mode = host_orch_func_ptr != nullptr;

    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO_V0(
        "RT2 bind: %d tensors + %d scalars, %s orchestration mode", tensor_count, scalar_count,
        host_orch_mode ? "host" : "device"
    );
    runtime->tensor_leases_.clear();

    int64_t t_total_start = _now_ms();

    ArenaSizingConfig sizing;
    if (!resolve_arena_sizing(ring_task_window, ring_heap, ring_dep_pool, &sizing)) {
        return -1;
    }

    // The retained temporary buffer is always used on this device path. It is an
    // internal allocation optimization, not user-facing config. Gate only on
    // whether the platform wired the slot accessors (a2a3 does; a backend
    // that leaves them null transparently falls back to per-tensor
    // device_malloc). The buffer itself lives on the runner across runs; here we
    // just grow it to this run's packed size and bump-slice from it.
    RetainedTempBump bump;
    bool use_temporary_buffer = !host_orch_mode && api->get_retained_temp_buffer != nullptr &&
                                api->set_retained_temp_buffer != nullptr;
    if (use_temporary_buffer && !bump.begin(api, orch_args)) {
        return -1;
    }

    auto bind_cleanup = RAIIScopeGuard([&]() {
        release_tensor_leases(runtime, api);
    });

    ChipStorageTaskArgs device_args;
    if (!stage_device_args(
            runtime, api, orch_args, signature, sig_count, use_temporary_buffer ? &bump : nullptr,
            host_orch_func_ptr != nullptr, &device_args
        )) {
        return -1;
    }


    int64_t t_prebuilt_start = _now_ms();
    {
        STRACE("simpler_run.bind.prebuilt");
        if (host_orch_mode) {
            ArenaStaticSizes sizes;
            if (!derive_arena_static_sizes(sizing, &sizes)) return -1;
            StaticArenaPtrs ptrs;
            if (!ensure_static_arenas(runtime, api, sizing, sizes, &ptrs)) return -1;
            if (!start_async_host_graph_pipeline(
                    runtime, api, sizing, sizes, ptrs, device_args, host_orch_func_ptr
                )) {
                return -1;
            }
        } else {
            PrebuiltRuntimeArenaCacheProbe cache_probe = make_prebuilt_runtime_arena_cache_probe(sizing);
            int cache_rc = bind_cached_runtime_image(runtime, api, cache_probe, device_args);
            if (cache_rc < 0) return -1;
            if (cache_rc != 0) {
                ArenaStaticSizes sizes;
                if (!derive_arena_static_sizes(sizing, &sizes)) return -1;

                StaticArenaPtrs ptrs;
                if (!ensure_static_arenas(runtime, api, sizing, sizes, &ptrs)) return -1;

                DeviceArena host_arena;
                PTO2RuntimeArenaLayout layout;
                if (!build_runtime_image(sizing, sizes, ptrs, &host_arena, &layout)) return -1;
                if (!bind_launch_state(runtime, api, ptrs, host_arena, layout, device_args)) return -1;
                store_prebuilt_runtime_image(runtime, api, cache_probe, ptrs, layout, host_arena);
            }
        }
    }
    int64_t t_prebuilt_end = _now_ms();

    LOG_INFO_V0(
        "%s orchestration ready: %d tensors + %d scalars", host_orch_mode ? "Host" : "Device", tensor_count,
        scalar_count
    );

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("TIMING: prebuilt_runtime_arena = %" PRId64 "ms", t_prebuilt_end - t_prebuilt_start);
    LOG_INFO_V0("TIMING: total_init_runtime_impl = %" PRId64 "ms", t_total_end - t_total_start);

    bind_cleanup.dismiss();
    return 0;
}

extern "C" void
configure_runtime_before_bind_impl(Runtime *runtime, const CallConfig *config, uint64_t callable_hash) {
    if (runtime == nullptr || config == nullptr) return;
    runtime->set_graph_cache_config(config->enable_graph_cache != 0, callable_hash);
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Releases recorded tensor leases
 * 3. Clears tensor lease state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }

    int rc = 0;

    if (join_async_host_graph_pipeline(runtime) != 0) {
        LOG_ERROR("HostGraph async orchestration failed");
        rc = -1;
    }

    LOG_INFO_V0("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorLease *tensor_leases = runtime->tensor_leases_.data();
    int tensor_lease_count = static_cast<int>(runtime->tensor_leases_.size());

    LOG_INFO_V0("Tensor leases to process: %d", tensor_lease_count);

    // PTO2 (device orchestration): graph output may be in packed buffer
    uint64_t graph_out_ptr = 0;
    uint64_t graph_out_size = 0;
    bool skip_tensor_copy_back = false;
    int32_t runtime_status = 0;
    PTO2SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    runtime_status = pto2_read_runtime_status(runtime, api, &host_header);
    if (runtime_status != 0) {
        int32_t orch_error_code = host_header.orch_error_code.load(std::memory_order_relaxed);
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_ERROR(
            "PTO2 runtime failed: orch_error_code=%d sched_error_code=%d runtime_status=%d", orch_error_code,
            sched_error_code, runtime_status
        );
        // A scheduler no-progress timeout (code 100) carries a device-classified
        // sub-reason + locators so the failure line is self-diagnosing without a
        // device-log dive. The full stall snapshot stays in the device log / plog.
        if (sched_error_code == PTO2_ERROR_SCHEDULER_TIMEOUT) {
            int32_t detail = host_header.sched_stall_detail.load(std::memory_order_acquire);
            LOG_ERROR(
                "PTO2 scheduler timeout sub_class=%s (detail=%d) completed=%d/%d running=%d ready=%d waiting=%d "
                "orch_done=%d stuck_task_id=%" PRId64 " stuck_core=%d",
                stall_detail_name(detail), detail, host_header.sched_stall_completed.load(std::memory_order_relaxed),
                host_header.sched_stall_total.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_running.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_ready.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_waiting.load(std::memory_order_relaxed),
                host_header.sched_stall_orch_done.load(std::memory_order_relaxed),
                host_header.sched_stall_task_id.load(std::memory_order_relaxed),
                host_header.sched_stall_core.load(std::memory_order_relaxed)
            );
        }
        skip_tensor_copy_back = true;
    } else {
        graph_out_ptr = host_header.graph_output_ptr;
        graph_out_size = host_header.graph_output_size;
        if (graph_out_ptr != 0) {
            LOG_INFO_V0("Graph output buffer: ptr=0x%" PRIx64 ", size=%" PRIu64, graph_out_ptr, graph_out_size);
        }
    }

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because PTO2 runtime reported fatal status");
    } else {
        bool first_output_tensor = true;
        for (int i = 0; i < tensor_lease_count; i++) {
            const TensorLease &lease = tensor_leases[i];

            // Skip if device pointer is null
            if (lease.dev_ptr == nullptr) {
                LOG_WARN("Tensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (lease.host_ptr == nullptr) {
                LOG_INFO_V0("Tensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still released through release_kind below.
            if (!lease.needs_copy_back) {
                LOG_INFO_V0("Tensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            void *src_ptr = lease.dev_ptr;
            size_t copy_size = lease.size;

            // Use graph_output_ptr for the first output tensor if available
            if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
                src_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(graph_out_ptr));
                copy_size = static_cast<size_t>(graph_out_size);
                LOG_INFO_V0("Using packed output buffer for tensor %d", i);
                first_output_tensor = false;
            }

            int copy_rc = api->copy_from_device(lease.host_ptr, src_ptr, copy_size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_INFO_V0("Tensor %d: %zu bytes copied to host", i, lease.size);
            }
        }
    }

    // Cleanup device tensors
    LOG_INFO_V0("=== Cleaning Up ===");
    release_tensor_leases(runtime, api);

    LOG_INFO_V0("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

    return rc;
}

// Extra AICPU entry symbols this runtime exports beyond the base
// {simpler_aicpu_exec, simpler_aicpu_init}. TMARB resolves orchestration on the
// device, so it exports simpler_aicpu_register_callable; the common AICPU loader
// queries this so it carries no runtime-specific symbol knowledge.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    static const char *const kExtra[] = {"simpler_aicpu_register_callable"};
    if (count != nullptr) {
        *count = sizeof(kExtra) / sizeof(kExtra[0]);
    }
    return kExtra;
}
