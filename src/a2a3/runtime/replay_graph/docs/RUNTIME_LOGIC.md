# replay_graph Runtime System Design

## Overview

`replay_graph` is a runtime for executing task graphs on Ascend AI
processors. It is derived from `tensormap_and_ringbuffer` but runs a
**single-shot, two-phase** model: the orchestrator builds the *whole* task
graph first, and only after it has fully finished do the schedulers start
dispatching. The two phases do not overlap, so there is no producer/consumer
flow control between them.

It coordinates four layers of execution:

- **Host** (x86/ARM CPU): compiles kernels, allocates device memory,
  initializes the runtime, and launches AICPU/AICore threads.
- **AICPU** (device ARM cores): runs the orchestrator (task graph builder)
  and the scheduler threads.
- **AICore** (AI compute cores): executes kernel functions dispatched by the
  schedulers.
- **Shared Memory** (Global Memory): one task ring, task descriptors, heap,
  and TensorMap shared between orchestrator and schedulers.

```text
┌───────────────────────────────────────────────────────────────────────┐
│                            Host (CPU)                                   │
│  test_*.py (SceneTestCase) → compile kernels → init Runtime             │
│  → upload binaries → launch AICPU/AICore → collect results              │
└───────────────────────────┬───────────────────────────────────────────┘
                            │ device memory / GM
┌───────────────────────────▼───────────────────────────────────────────┐
│                     AICPU (4 threads)                                   │
│  Phase 1 — Thread 3: Orchestrator builds the whole task graph,          │
│            submit_task wires fanout lists + initial_ready[] in-line      │
│  Phase 2 — Threads 0-2: Schedulers seed initial_ready[] into their      │
│            ready queues, then dispatch + handle completion              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Shared Memory (GM)                             │   │
│  │  SharedMemoryHeader │ TaskDescriptors[] │ Payloads │ SlotStates  │   │
│  │  GM Heap (output buffers)                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Scheduler ──Handshake/Registers──► AICore workers (AIC + AIV)          │
└───────────────────────────────────────────────────────────────────────┘
```

The defining differences from `tensormap_and_ringbuffer`:

- **Two non-overlapping phases**: orchestration runs to completion, then
  scheduling runs. Not a pipelined producer/consumer.
- **Single ring**: one task ring / heap / dep pool. No per-scope ring array.
- **Wiring owned by the orchestrator**: the fanout dependency graph is built
  during the orch phase; schedulers only read it.
- **Pure bump allocation**: the arena fills exactly once and never wraps.
  Overflow is a fatal sizing error, not back-pressure.
- **Two-state task lifecycle**: `PENDING → COMPLETED` (terminal). There is no
  `CONSUMED` state, no slot reclaim, and no watermark advancement.

---

## 1. Single-Shot Execution Model

`replay_graph` assumes the entire task graph fits in the configured window:
total tasks ≤ task window, total output bytes ≤ heap, total fanout edges ≤
dep pool. Under that assumption, the runtime drops every reclamation and
back-pressure mechanism that `tensormap_and_ringbuffer` needs:

| Concern | tensormap_and_ringbuffer | replay_graph |
| ------- | ------------------------ | ------------ |
| Phasing | Orchestrator and schedulers run concurrently | Orchestrator fully precedes schedulers |
| Rings | `PTO2_MAX_RING_DEPTH` per-scope rings | One ring, `ring_id` is always 0 |
| Wiring | Scheduler thread 0 drains a wiring queue | Orchestrator builds fanout lists in-line in `submit_task` |
| Allocation | Ring wrap + back-pressure spin | Pure bump; overflow is fatal |
| Task states | `PENDING → COMPLETED → CONSUMED` | `PENDING → COMPLETED` (terminal) |
| Reclaim | `advance_ring_pointers` frees slots/heap | None; the arena is never reclaimed |
| Exit | All tasks consumed + orchestrator done | `completed_tasks >= total_tasks` |

---

## 2. Platform Abstraction

Two platform implementations exist under `src/a2a3/platform/`, sharing a
common interface.

### 2.1 a2a3 (Real Ascend Hardware)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses CANN APIs: `rtMalloc`, `rtMemcpy`, `rtLaunchKernel` |
| `memory_allocator.cpp` | Wraps `rtMalloc`/`rtFree` with allocation tracking |
| `aicore/kernel.cpp` | `KERNEL_ENTRY(aicore_kernel)` → `aicore_execute` |
| `aicpu/kernel.cpp` | `DynTileFwkBackendKernelServer` entry → `aicpu_execute` |
| `spin_hint.h` | ARM `wfe`/`yield` instructions for efficient spinning |

### 2.2 a2a3sim (Thread Simulation)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses `std::thread` to simulate AICPU/AICore |
| `memory_allocator.cpp` | Wraps `malloc`/`free` |
| `aicore/kernel.cpp` | `aicore_execute_wrapper` sets `g_sim_reg_base` per core |
| `upload_chip_callable_buffer` | Copy ChipCallable bytes to a host scratch, `dlopen` each child SO, `dlsym` "kernel_entry", patch `resolved_addr_` |

### 2.3 Platform Constants (`platform_config.h`)

| Constant | Value | Description |
| -------- | ----- | ----------- |
| `PLATFORM_MAX_BLOCKDIM` | 24 | Maximum blocks (each = 1 AIC + 2 AIV) |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU thread count (3 schedulers + 1 orchestrator) |
| `PLATFORM_MAX_AIC_PER_THREAD` | 24 | Max AIC cores per scheduler thread |
| `PLATFORM_MAX_AIV_PER_THREAD` | 48 | Max AIV cores per scheduler thread |
| `PLATFORM_PROF_SYS_CNT_FREQ` | 50 MHz | System counter frequency for profiling |

---

## 3. Shared Memory Layout

The orchestrator and schedulers communicate through one contiguous shared
memory region in Global Memory (GM). There is a single ring, so flow control,
layout metadata, and data pointers live flat in `PTO2SharedMemoryHeader`.

```text
┌─────────────────────────────┐  offset 0
│  PTO2SharedMemoryHeader     │  (single-ring flow control + layout, flags)
├─────────────────────────────┤  aligned
│  PTO2TaskDescriptor[N]      │  N = task_window_size
│  PTO2TaskPayload[N]         │
│  PTO2TaskSlotState[N]       │
└─────────────────────────────┘
```

### 3.1 SharedMemoryHeader Fields

| Field | Writer | Reader | Purpose |
| ----- | ------ | ------ | ------- |
| `task_count` | Orchestrator | Scheduler | Total tasks submitted (frozen after orch; equals next task ID during orch) |
| `orchestrator_done` | Orchestrator | Scheduler | Gates the scheduler exit check |
| `task_window_size` | Init | Both | Number of task slots |
| `heap_size` | Init | Both | Heap total size |
| `task_descriptors_offset` | Init | Both | Offset to the TaskDescriptor array in SM |
| `total_size` | Init | Both | Total shared memory size |
| `graph_output_ptr` | Orchestrator | Host | Address of final output (packed buffer) |
| `graph_output_size` | Orchestrator | Host | Size of final output in bytes |
| `orch_error_code` | Orchestrator | Scheduler/Host | Fatal-error channel |

### 3.2 Size Calculation

```text
total = ALIGN(Header)
      + ALIGN(window_size * sizeof(TaskDescriptor))
      + ALIGN(window_size * sizeof(TaskPayload))
      + ALIGN(window_size * sizeof(TaskSlotState))
```

Alignment is 64 bytes (`PTO2_ALIGN_SIZE`).

---

## 4. Allocation Mechanisms (Pure Bump)

There is exactly one of each allocator. None of them wrap or reclaim: the
orchestrator fills each arena once during the orch phase, and the schedulers
(which run only afterward) never free anything. Running out of any resource
is a **fatal sizing error**, reported through `orch_error_code` — never a
spin-wait.

### 4.1 Task Allocator (`PTO2TaskAllocator`)

A unified task-slot + heap-buffer bump allocator. The orchestrator is single
threaded, so it tracks both the next task id and the heap top locally and
publishes `task_count` with a plain release store — no CAS.

```text
alloc(output_size):
    if local_task_id + 1 >= window_size:      # window full
        report_task_overflow(); return failed
    if heap_top + ALIGN(output_size) > heap_size:   # heap full
        report_heap_overflow(); return failed
    p = heap_base + heap_top
    heap_top += ALIGN(output_size)
    task_id = local_task_id++
    task_count.store(local_task_id)   # release
    return {task_id, slot = task_id, p, p + ALIGN(output_size)}
```

- **Slot mapping**: total tasks ≤ window, so `slot == task_id` — the modulo
  (`task_id & (window_size - 1)`) never wraps.
- **Overflow**: window exhaustion sets `PTO2_ERROR_FLOW_CONTROL_DEADLOCK`,
  heap exhaustion sets `PTO2_ERROR_HEAP_RING_DEADLOCK`. Both are fatal — fix
  by enlarging `PTO2_TASK_WINDOW_SIZE` / `PTO2_HEAP_SIZE` (or the matching
  `PTO2_RING_*` env), not by waiting.
- **State queries**: `active_count() == local_task_id` (everything allocated
  is still live); `task_tail()` and `heap_tail()` stay at 0.

### 4.2 Dependency List Pool (`PTO2DepListPool`)

A bump allocator for fanout linked-list nodes. `submit_task` builds the
fanout lists here in-line as each task is submitted during the orch phase;
the schedulers only traverse them read-only during completion. Capacity must
hold every fanout edge of the entire graph, and overflow is fatal.

- **Entry 0**: NULL sentinel.
- **Allocation**: `top++`, never wraps.
- **Reclamation**: none — `tail` is fixed, `used() == top - tail`.

### 4.3 No Back-Pressure, No Deadlock Path

Because the schedulers never run during the orch phase, there is nothing to
wait for: an allocator that cannot satisfy a request can never make progress
by spinning, so it fails fatally instead. This eliminates the entire
flow-control / deadlock-detection machinery of
`tensormap_and_ringbuffer` — there are no `BLOCKED` warnings, no spin
counters, and no circular scope/`scope_end` deadlock.

**Sizing guideline**: each window must be ≥ the *whole graph's* demand, not
just the largest scope. `task_window_size ≥ total_tasks`,
`heap_size ≥ Σ output bytes`, `dep_pool ≥ total fanout edges`.

---

## 5. TensorMap and Automatic Dependency Tracking

### 5.1 Purpose

TensorMap maps tensor memory regions to their producer task IDs. When a new
task reads a tensor (INPUT/INOUT), TensorMap discovers the producer and
establishes a dependency edge.

### 5.2 Hash Table Design

- **Key**: tensor base address (`buffer.addr`).
- **Value**: producer task ID, with overlap detection for sub-regions.
- **Overlap**: `COVERED` (new region fully contains old) or `OTHER`
  (partial overlap).
- Sub-tensors of the same base tensor hash to the same bucket.

### 5.3 Entry Pool (Bump-Only)

TensorMap entries are allocated from a fixed pool by bump (`entry_pool[
next_entry_idx++]`). In the single-shot model there is no retirement: the
orchestrator never retires a task, so no entry is ever freed
during the orch phase, and the schedulers never touch the map. The pool is
bounded by `PTO2_TENSORMAP_POOL_SIZE` (65536) and must hold every entry the
graph produces; exhaustion is fatal.

There is no free list, no `cleanup_retired`, and no pool back-pressure —
those existed in `tensormap_and_ringbuffer` only to recycle entries of
retired tasks.

### 5.4 Dependency Discovery Flow

When `PTO2OrchestratorState::submit_task` processes parameters:

1. **INPUT/INOUT**: `PTO2TensorMap::lookup` searches for overlapping
   producers; each found producer is recorded as a fanin via
   `append_fanin_or_fail`.
2. **OUTPUT/INOUT**: `PTO2TensorMap::insert` registers the current task as
   the new producer at the bucket head.

Lookup walks the bucket chain in descending task-id order; since nothing is
retired, every entry is valid and no chain truncation is needed.

---

## 6. Task Descriptor and States

### 6.1 PTO2TaskDescriptor (Hot Path)

| Field | Description |
| ----- | ----------- |
| `task_id` | `PTO2TaskId` (64-bit: `ring_id << 32 \| local_id`). `ring_id` is always 0; see §6.3. |
| `kernel_id[3]` | Per-slot kernel IDs `[AIC, AIV0, AIV1]`; `INVALID_KERNEL_ID` = inactive |
| `active_mask` | Active subtask slots: `bit0=AIC`, `bit1=AIV0`, `bit2=AIV1` |
| `completed_subtasks` | Atomic counter; trigger condition: `completed_subtasks == total_required_subtasks` |
| `fanin_count` | Number of producer dependencies + 1 sentinel (set during wiring) |
| `fanout_head` | Head of the fanout consumer list (pointer; written only by orch wiring, read-only in sched) |
| `packed_buffer_base` | Start of packed output buffer in GM heap |
| `packed_buffer_end` | End of packed output buffer |

There is no `fanout_count` / `fanout_refcount` field: the single-shot model
removed the consumer-counting that drove the dropped `CONSUMED` lifecycle.

### 6.1b PTO2TaskPayload (Cold Path)

| Field | Description |
| ----- | ----------- |
| `tensors[]` | Tensor descriptors for parameters |
| `scalars[]` | Scalar parameter values |
| `tensor_count` / `scalar_count` | Number of valid tensor / scalar parameters |
| `dispatch_fanin` | Early-dispatch counter (consumer-side); seeded by `submit_task` |
| `staged_core_mask[]` / `spec_*` | Speculative early-dispatch metadata |

The payload no longer stores the producer (fanin) list: `submit_task` builds
the fanout graph in-line via the slot's `fanout_head` / `fanin_count`, so the
producer edges live in `dep_pool`, not the payload.

### 6.2 Task State Machine

```text
  [0] PENDING ──worker(s) done──► [1] COMPLETED (terminal)
```

In `PTO2TaskSlotState::task_state` (`std::atomic<PTO2TaskState>`):

- **0 (PENDING)**: slot is allocated and stays PENDING through "waiting on
  producers", "queued in a ready queue", and "dispatched to a worker"; the
  ready/running distinction is derived from `fanin_refcount` and per-core
  `running_slot_state`.
- **1 (COMPLETED)**: hardware execution complete. This is terminal — there
  is no `CONSUMED` state, no slot reclaim, and the slot is never reused.

### 6.3 Task ID Encoding (single ring)

`PTO2TaskId` packs a ring id and a local id into 64 bits
(`(ring_id << 32) | local_id`) for compatibility with the descriptor and
TensorMap entry types. In `replay_graph` there is only one ring, so
`ring_id` is always 0 and `task_id.local()` is the canonical slot index.
`PTO2TaskSlotState::ring_id` likewise stays 0. The encoding is retained
rather than narrowed so the shared submit/dispatch types match
`tensormap_and_ringbuffer`.

---

## 7. Orchestrator (Phase 1)

### 7.1 PTO2OrchestratorState

The orchestrator runs on AICPU Thread 3 and builds the task graph by calling
the user-provided orchestration function. Key members:

- `task_allocator`, `dep_pool`: the single-shot bump allocators (task slots +
  heap, fanout dependency lists).
- `tensor_map`: producer lookup.
- `fanin_seen_epoch[]`: per-slot dedup stamp so a producer counted twice for
  one consumer (e.g. duplicate explicit dep) yields a single fanin edge.
- `scope_stack_top`, `manual_begin_depth`: scope nesting depth + manual-scope
  bookkeeping (no per-scope task lists — slot reclaim was dropped).
- `initial_ready[]`, `initial_ready_count`: the handoff array of tasks whose
  fanin is already satisfied when `submit_task` returns, consumed by the
  schedulers.
- `gm_heap_base`, `gm_heap_size`: GM heap for output buffers.

### 7.2 Task Submission Flow (`submit_task`)

| Step | Operation |
| ---- | --------- |
| 1 | `PTO2TaskAllocator::alloc` — bump a task slot + heap buffer (fatal on overflow); bind the slot's payload/task pointers |
| 2 | **Wire fanin**: for each explicit dep and each INPUT/INOUT producer found in TensorMap, call `append_fanin_or_fail` to build the graph edge in-line (see §7.3) |
| 3 | **Insert**: register OUTPUT/INOUT args in TensorMap |
| 4 | **Materialize payload**: `payload.init` copies tensors/scalars and zeroes the early-dispatch counters |
| 5 | **Seed initial-ready**: if `fanin_count == 0` (no pending-producer edges remain), append the task to `initial_ready[]` |

`submit_task` builds the whole dependency graph as it runs — there is no
separate wiring pass after orchestration. The fanin loop never touches the
payload (already-complete producers are not counted), so `payload.init` simply
runs after registration; the scheduler later seeds `dispatch_fanin` via
`propagate_dispatch_fanin`.

### 7.3 In-line Wiring (`append_fanin_or_fail`)

For each distinct producer of the task being submitted, `append_fanin_or_fail`
wires one graph edge. Producers are deduped per-submit via `fanin_seen_epoch`
(a duplicate producer returns early). For a fresh producer:

1. If the producer is already `>= COMPLETED` (only inline-completed alloc
   tasks during the orch phase), the dependency is already satisfied, so it is
   **not counted at all** — `fanin_count` / `fanin_refcount` / `dispatch_fanin`
   track only pending producers. No fanout edge is built either (such a
   producer never fires `on_task_complete`).
2. Otherwise the producer is still PENDING: `consumer->fanin_count++` (one
   pending producer edge), then prepend this consumer to the producer's
   `fanout_head` via `dep_pool.prepend`. No lock is taken — wiring is
   single-threaded and runs entirely before the scheduler starts, so
   `fanout_head` has no concurrent reader (the orch→sched barrier is §8.2).
   `dep_pool` is never reclaimed during the orch phase, so its capacity must
   hold every fanout edge; overflow is fatal.

After the fanin loop, `submit_task` checks `fanin_count == 0`: a task with no
pending producers — either it had no producers at all, or every producer was
an already-complete alloc (none counted) — is appended straight to
`initial_ready[]`.

### 7.4 Scope Mechanism (`PTO2_SCOPE`)

Scopes track nesting depth (`scope_stack_top`) and manual-scope semantics
(`manual_begin_depth`) only. In the single-shot model `scope_end` reclaims
nothing — it just pops the depth and resets manual bookkeeping; tasks and
their buffers live for the whole run.

```cpp
PTO2_SCOPE(rt) {
    rt_submit_aic_task(FUNC_QK, args);
    rt_submit_aiv_task(FUNC_SF, args);
}
```

**Output tensor lifetime.** `submit_task` returns a `TaskOutputTensors`, and
`get_ref(i)` hands back a `const Tensor&` backed by the submitting task's
`PTO2TaskPayload::tensors[]`. Because slots are never reused in the
single-shot model, that storage stays valid for the whole run — but the
contract is unchanged from `tensormap_and_ringbuffer`: treat the references
as scoped to the submitting `PTO2_SCOPE` and do not let them escape.

---

## 8. Scheduler (Phase 2)

### 8.1 Thread Model

With `aicpu_thread_num=4`, the AICPU runs 4 threads:

| Thread | Role | Cores |
| ------ | ---- | ----- |
| 0 | Scheduler | 6 AIC + ~13 AIV |
| 1 | Scheduler | 6 AIC + ~13 AIV |
| 2 | Scheduler | 6 AIC + ~13 AIV |
| 3 | Orchestrator | none |

AICs and AIVs are divided equally among the 3 scheduler threads. After
orchestration, the orchestrator thread's cores are reassigned so all
hardware is available for dispatch.

### 8.2 Phase Handoff

The schedulers start only once the orchestrator publishes
`orchestrator_done`. Before dispatching, one scheduler thread wins a
one-time seed claim and pushes the orchestrator's `initial_ready[]` array
into the per-shape ready queues (`push_ready_routed`). After that, the
schedulers run their dispatch + completion loop.

### 8.3 Scheduler Main Loop

Each scheduler thread runs a tight loop with two phases:

**Completion handling**:

- Poll register `COND` on each managed core.
- On `TASK_FIN`: record timestamps and call `on_subtask_complete`; when
  `completed_subtasks == total_required_subtasks`, call `on_task_complete`,
  which:
  1. Stores `task_state = COMPLETED`.
  2. Reads `fanout_head` directly (lock-free): it was frozen by the orch
     phase (built in-line by `submit_task`) before `orchestrator_done_` was
     released, so there is no
     concurrent writer. The per-task spinlock that previously guarded it was
     dropped in the single-shot model — it MUST be restored before any
     orch/sched time-overlap (stage-3 ping-pong).
  3. Walks the fanout list, releasing one fanin from each consumer
     (`release_fanin_and_check_ready`); a consumer whose `fanin_refcount`
     reaches `fanin_count` is pushed to its ready queue.

  There is no `CONSUMED` transition, no `fanout_refcount`, no
  `advance_ring_pointers`, and no slot/heap reclaim.

**Dispatch**:

- For each idle core: pop a task from the matching shape-based ready queue
  (lock-free MPMC Vyukov queue, one per resource shape).
- Build `PTO2DispatchPayload` from the `TaskDescriptor`.
- Write the task pointer to `Handshake.task`, signal the AICore via
  register `DATA_MAIN_BASE`.

### 8.4 Exit Condition

A scheduler thread exits when the orchestrator is done **and** every task
has completed:

```text
if (!orchestrator_done) return;            # gate: graph not finished
if (total_tasks > 0 &&
    completed_tasks >= total_tasks) {      # all work done
    completed = true; break;
}
```

`completed_tasks` counts COMPLETED tasks (plus the orchestrator's
`inline_completed_tasks`, folded in at handoff). Termination is purely a
COMPLETED count reaching `total_tasks`; `orchestrator_done` is only the gate
that makes the count meaningful.

### 8.5 Ready Queue Design

Ready queues use a lock-free bounded MPMC (Vyukov) design:

- One `PTO2ReadyQueue` per resource shape (`AIC_ONLY`, `AIV_X1`, `AIV_X2`,
  `AIC_AIV_X1`, `AIC_AIV_X2`).
- **Push**: the seed pass (initial-ready) or a scheduler on completion
  pushes newly-ready tasks to the queue matching
  `task->active_mask.to_shape()`.
- **Pop**: scheduler threads pop from the queue matching an idle core's
  resource shape.
- Per-slot sequence counters prevent ABA; `enqueue_pos` / `dequeue_pos` are
  on separate cache lines.

### 8.6 SchedulerContext

All scheduler-side state and methods live in `SchedulerContext`
(`runtime/scheduler/scheduler_context.h`), held as `sched_ctx_` of
`AicpuExecutor`. `AicpuExecutor` owns the lifecycle atomics and the
orchestration SO handle and delegates the rest.

Public surface (called from `AicpuExecutor::init/run/deinit`):

| Method | Phase | Purpose |
| ------ | ----- | ------- |
| `init(...)` | once per run | Handshake + assign cores, reset counters, latch `regs_base`, bind `func_id_to_addr_` |
| `bind_runtime(rt)` | device-orch only | Wire `sched_` to `rt->scheduler` once the orchestrator creates `rt` |
| `resolve_and_dispatch(runtime, thread_idx)` | per scheduler thread | Main dispatch loop |
| `shutdown(thread_idx)` | per thread on exit | `platform_deinit_aicore_regs`; PMU finalize when enabled |
| `on_orchestration_done(...)` | orchestrator thread | Publish core assignments, latch task count, fold inline-completed tasks, flip `orchestrator_done_`, drive orch→sched core transition |
| `deinit()` | once per run | Reset scheduler-owned fields |
| Read-only accessors | various | `is_completed()` / `completed_tasks_count()` / `wait_pto2_init_complete()` |

Private internals are split across three .cpp files:

- `scheduler_completion.cpp` — completion polling, drain protocol
- `scheduler_dispatch.cpp` — dispatch loop, initial-ready seeding
- `scheduler_cold_path.cpp` — exit checks, diagnostics, profiling,
  lifecycle, core management, `on_orchestration_done`

---

## 9. AICore Worker Interaction

### 9.1 Handshake Protocol

Each AICore worker has a `Handshake` struct in shared memory:

| Field | Direction | Purpose |
| ----- | --------- | ------- |
| `task` | AICPU→AICore | Pointer to `PTO2DispatchPayload` |
| `control` | AICPU→AICore | 0=normal, 1=shutdown |
| `perf_records_addr` | AICPU→AICore | Performance buffer address |

### 9.2 Register-Based Dispatch

The production protocol uses hardware registers rather than a shared-memory
status flag. `task_id` is 64-bit but the registers are 32-bit, so a per-core
monotonic dispatch counter (`s_dispatch_seq`) is written to `DATA_MAIN_BASE`
to give every dispatch a unique value (identical values would be read as a
stale dispatch).

| Register | Direction | Usage |
| -------- | --------- | ----- |
| `DATA_MAIN_BASE` | AICPU→AICore | Per-core dispatch sequence (idle=0x7FFFFFFD); `EXIT_SIGNAL` to shut down |
| `COND` | AICore→AICPU | `[bit31=state, bits30:0=seq]`: ACK (state=0) or FIN (state=1) |

**AICore execution loop**:

1. Poll `DATA_MAIN_BASE` for a value != AICPU_IDLE_TASK_ID.
2. Read the payload from `Handshake.task`.
3. Write ACK to `COND`.
4. Execute the kernel via `func_id_to_addr` lookup.
5. Write FIN to `COND`.

### 9.3 PTO2DispatchPayload

Built by the scheduler from `PTO2TaskDescriptor`:

| Field | Description |
| ----- | ----------- |
| `task_id` | Task identifier (for completion aggregation) |
| `subslot` | Which subtask slot this dispatch represents (`AIC`, `AIV0`, `AIV1`) |
| `kernel_id` | Function ID for this subtask slot |
| `core_type` | AIC or AIV |
| `function_bin_addr` | GM address of the compiled kernel binary |
| `num_args` | Number of arguments |
| `args[]` | Tensor addresses and scalar values |

---

## 10. Kernel and Orchestration Loading

### 10.1 Kernel Binary Loading

1. **Host** compiles each kernel source (`.cpp`) into a binary and packs all
   children into a single `ChipCallable` buffer alongside the orchestration
   SO.
2. `host_api.upload_chip_callable_buffer(callable)` H2Ds the whole buffer
   once and returns the device address of the ChipCallable header.
3. For each child, host computes
   `chip_dev + offsetof(ChipCallable, storage_) + callable->child_offset(i)`
   and stores it in `Runtime.func_id_to_addr_[child_func_id(i)]`.
4. When dispatching, the scheduler reads `func_id_to_addr_[fid]`, reads
   `resolved_addr_`, and copies it into
   `PTO2DispatchPayload.function_bin_addr`.

### 10.2 Orchestration SO Loading

1. **Host** compiles the orchestration source into a shared library.
2. The SO is embedded into `Runtime.device_orch_so_storage_[]` and copied to
   device.
3. **AICPU Thread 3** writes the SO to a temp file and calls `dlopen`.
4. `dlsym("aicpu_orchestration_config")` returns configuration.
5. `dlsym("aicpu_orchestration_entry")` returns the orchestration function.
6. Thread 3 creates a `PTO2Runtime` and calls the orchestration function
   within a `PTO2_SCOPE`; each `submit_task` builds its graph edges in-line.
7. After orchestration: `dlclose`, delete the temp file.

### 10.3 Thread Startup Synchronization

| Flag | Set by | Waited by | Purpose |
| ---- | ------ | --------- | ------- |
| `runtime_init_ready_` | Thread 3 | Threads 0-2 | Runtime and SM handle initialized |
| `pto2_init_claimed_` | First init thread | Others | One-time memset of arrays started |
| `pto2_init_complete_` | Init thread | Thread 3 + others | One-time init of per-task arrays done |

Startup sequence:

1. Thread 3: create SM handle + runtime → set `runtime_init_ready_`.
2. Scheduler threads: wait for `runtime_init_ready_` → one thread wins
   `pto2_init_claimed_` → memset per-task arrays → set
   `pto2_init_complete_`; others wait.
3. Thread 3: wait for `pto2_init_complete_` → configure
   orchestrator/scheduler pointers.
4. Thread 3: run orchestration (graph wired in-line by `submit_task`) → set
   `orchestrator_done_`.
5. Scheduler threads: seed `initial_ready[]`, then enter the dispatch loop.

---

## 11. Orchestration API

The orchestration API is defined in `pto_orchestration_api.h`. Orchestration
code depends only on this header.

### 11.1 Core API

| Function/Macro | Purpose |
| -------------- | ------- |
| `rt_submit_task(mixed_kernels, args)` | Submit a mixed task with `MixedKernels` |
| `rt_submit_aic_task(kernel_id, args)` | Convenience: submit AIC-only task |
| `rt_submit_aiv_task(kernel_id, args)` | Convenience: submit AIV-only task |
| `PTO2_SCOPE() { ... }` | RAII scope for task grouping |
| `rt_orchestration_done()` | Signal orchestration complete |

### 11.2 Parameter Construction

| Function | Description |
| -------- | ----------- |
| `make_tensor_external(ptr, shapes, ndim, dtype)` | Wrap an existing device pointer as a tensor |
| `TensorCreateInfo(shapes, ndim, dtype)` | Describe a runtime-created output buffer |
| `Arg::add_input(tensor)` | INPUT parameter — read by the task |
| `Arg::add_output(create_info)` | OUTPUT parameter — runtime allocates and returns a Tensor |
| `Arg::add_inout(tensor)` | INOUT parameter — existing tensor read then written |
| `Arg::add_scalar(value)` | 64-bit scalar parameter |

### 11.3 Resource Shapes

Tasks are queued by resource shape, derived from `active_mask`:

| Shape | Active Mask | Description |
| ----- | ----------- | ----------- |
| `AIC_ONLY` | AIC only | AIC cores (matrix multiplication) |
| `AIV_X1` | AIV0 or AIV1 only | Single AIV core (vector operations) |
| `AIV_X2` | AIV0 + AIV1 | Two AIV cores |
| `AIC_AIV_X1` | AIC + one AIV | AIC + single AIV core |
| `AIC_AIV_X2` | AIC + AIV0 + AIV1 | Full cluster (AIC + two AIV cores) |

### 11.4 Orchestration Export Interface

Each orchestration `.so` must export:

```cpp
extern "C" PTO2OrchestrationConfig
aicpu_orchestration_config(uint64_t *args, int arg_count);
extern "C" void aicpu_orchestration_entry(uint64_t *args, int arg_count);
```

---

## 12. Example: Paged Attention

### 12.1 Test Configuration

```python
@scene_test(level=2, runtime="replay_graph")
class PagedAttentionUnroll(SceneTestCase):
    ...
```

The scene test selects the runtime with `runtime="replay_graph"` and lists
kernel sources + the orchestration source under its `kernels/` directory.

### 12.2 Orchestration Structure

```cpp
void aicpu_orchestration_entry(uint64_t *args, int arg_count) {
    // Unpack args: query, key_cache, value_cache, ..., out, config
    for (q_idx = 0; q_idx < q_loop; q_idx++) {
        for (batch_start = 0; batch_start < batch; batch_start += STEP) {
            PTO2_SCOPE() {
                // Submit AIV_HUB to initialize accumulators
                for (bn = 0; bn < max_bn; bn++) {
                    // Submit QK (CUBE) → SF (VECTOR) → PV (CUBE) → UP (VECTOR)
                }
            }
        }
    }
}
```
