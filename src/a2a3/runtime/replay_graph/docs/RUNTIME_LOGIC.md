# Replay Graph Runtime Logic

## 1. Execution Contract

`replay_graph` is a single-shot, two-phase runtime:

1. One AICPU orchestrator executes the orchestration function to completion.
2. Every task and dependency edge is materialized in a single graph arena.
3. The orchestrator release-stores `orchestrator_done`.
4. Scheduler threads acquire the barrier, seed the frozen graph's initially
   ready tasks, and begin dispatch.

The scheduler never overlaps graph construction. A replay therefore has one
ring identity (`ring_id == 0`) but no ring behavior: task slots, heap bytes,
dependency entries, and TensorMap entries are allocated monotonically and are
not reclaimed or reused until the next replay.

This removes online wiring, wrap handling, watermarks, lifecycle retirement,
and the synchronization they required. In exchange, configured capacities
must hold the complete graph.

## 2. Lifecycle

Task state is monotonic:

```text
PENDING -> COMPLETED
```

There is no `CONSUMED` state. Completion does not release producers, retire
TensorMap entries, advance a tail, or reset a slot. Scopes preserve nesting,
manual-dependency, DFX, and profiling semantics; they do not own resource
lifetimes.

The pooled runtime arena may be reused by a later invocation. The AICPU resets
shared-memory slots, bump allocators, ready-queue run state, and TensorMap epoch
state before the next orchestration begins. This between-replay reset is not
in-replay reclamation.

## 3. Memory Ownership

### Shared memory

`PTO2SharedMemoryHeader` owns the whole-graph communication image:

- `fc.task_count`: number of submitted tasks, frozen before scheduling
- `task_descriptors[]`, `task_payloads[]`, `slot_states[]`
- `orchestrator_done`: release/acquire phase barrier
- graph output metadata and fatal/stall diagnostics

Task id and slot are identical for the lifetime of a replay. There is no modulo
slot mapping.

### Orchestrator arena

The orchestrator owns all graph-construction storage:

- `PTO2TaskAllocator`: task id and heap bump pointers
- `PTO2DepListPool`: every frozen producer-to-consumer fanout edge
- `fanin_seen_epoch[]`: per-submit producer deduplication
- `initial_ready[]`: tasks whose exact `fanin_count` is zero
- `PTO2TensorMap`: automatic tensor dependency discovery

### Scheduler arena

The scheduler owns only execution-time structures such as ready queues,
early-dispatch queues, async wait state, and profiling counters. It does not
own dependency storage and cannot mutate graph topology.

## 4. Allocation

`PTO2TaskAllocator` is a pure bump allocator. Each successful allocation:

1. reserves the next task id and same-numbered slot;
2. aligns output storage to `PTO2_ALIGN_SIZE`;
3. advances the heap top;
4. release-publishes the new task count.

It never spins, wraps, observes scheduler progress, or moves a tail. Task-window
or heap exhaustion is a fatal sizing error. `PTO2DepListPool` and TensorMap use
the same whole-graph capacity contract.

The effective capacities come from RuntimeEnv slot 0, `PTO2_RING_*` environment
variables, or compile-time defaults. The legacy names remain for command-line
compatibility; comma-separated per-ring values are invalid for this runtime.

## 5. Submit And Wiring

`submit_task` performs graph construction inline:

1. Allocate and bind the task descriptor, payload, slot state, and output
   storage.
2. Start a fresh `fanin_seen_epoch` for producer deduplication.
3. For each explicit dependency, validate that it names an earlier ring-0 task
   and call `append_fanin_or_fail`.
4. Query TensorMap for creator and overlapping modifier dependencies and call
   the same append helper for each result.
5. For each unique pending producer, prepend one consumer entry to the
   producer's frozen `fanout_head` and increment the consumer's exact
   `fanin_count`.
6. Initialize task payload and scheduling metadata.
7. Insert outputs and inout modifiers into TensorMap.
8. If `fanin_count == 0`, append the slot to `initial_ready[]`.

There is no wiring queue, fanin storage in the payload, fanin builder, fanout
lock, or `+1` sentinel. A synchronously completed hidden allocation producer
does not contribute to `fanin_count` and does not receive a fanout edge.

Explicit and TensorMap-derived references to the same producer are deduplicated
by slot for that submit. Dependency pool use is therefore exactly the number of
unique pending producer-to-consumer edges.

## 6. Graph Freeze Barrier

At orchestration completion, `mark_done()` logs final capacity use and performs:

```cpp
orchestrator_done.store(1, std::memory_order_release);
```

Scheduler threads wait with an acquire load. One scheduler then publishes all
`initial_ready[]` slots into routed ready queues. The once-guard is reset between
replays. After the acquire, scheduler workers see immutable fanout lists, exact
fanin counts, initialized payloads, and the final task count.

This ordering is why fanout writes need no lock and why wiring needs no
sentinel. Any future design that overlaps orchestration and scheduling must
restore an equivalent graph-publication synchronization protocol.

## 7. Scheduling And Completion

For a normal completion, the scheduler:

1. marks the producer `COMPLETED`;
2. traverses its immutable `fanout_head`;
3. atomically increments each consumer's `fanin_refcount`;
4. routes a consumer exactly once when
   `fanin_refcount == fanin_count`.

The scheduler exits based on completed task count, including synchronously
completed hidden allocation tasks accounted by the orchestrator. It does not
wait for a consumer lifecycle or drain a reclaim queue.

Dummy tasks have no active AICore mask. They enter the dummy ready queue and
complete inline, propagating dependencies without launching a kernel.

## 8. Early Dispatch

Latest-main early-dispatch and sync-start scheduling are retained, but they run
only after the graph-freeze barrier.

`dispatch_fanin` is not seeded by `submit_task`. It starts at zero and is
advanced by scheduler propagation when a producer's launch payload is fully
published. A consumer becomes an early-dispatch candidate when
`dispatch_fanin == slot_state.fanin_count`. This counter is an execution-time
launch-readiness signal; `fanin_refcount` remains the completion-readiness
signal.

The graph itself remains immutable while these counters and queue states evolve.

## 9. TensorMap

TensorMap stores producer metadata for automatic RAW/WAW dependency discovery.
Entries remain valid for the entire replay. Lookup does not test a retirement
watermark, and no per-task TensorMap chain or cleanup cursor exists.

Bucket epochs allow the pooled TensorMap allocation to be reset cheaply between
replays. Within one graph, capacity must accommodate all live entries. Pool
exhaustion is fatal rather than a request to reclaim completed tasks.

## 10. Scalar Data Access

`get_tensor_data` and `set_tensor_data` can wait for producer completion, but
the single-shot lifecycle cannot wait for consumer retirement. Consequently,
`set_tensor_data` provides WAW protection only, not WAR protection.

Because scheduler dispatch begins after orchestration finishes, orchestration
must not synchronously read data produced by a task submitted in the same
replay. Such dynamic data-dependent orchestration belongs to an overlapping
runtime, not this two-phase model.

## 11. Capacity Failures

The following indicate an undersized complete-graph arena and fail fast:

- task window overflow;
- heap overflow;
- dependency pool overflow;
- TensorMap pool overflow;
- initial-ready handoff overflow.

Sizing should include all hidden allocation tasks, output buffers, unique
dependency edges, and TensorMap producer entries generated by the orchestration.

## 12. Invariants

- `ring_id == 0` for every task.
- `slot == task_id.local()` and a slot is initialized once per replay.
- Orchestration finishes before any scheduler dispatch.
- Only the orchestrator writes graph topology.
- Scheduler workers only read fanout topology after the acquire barrier.
- `fanin_count` is the exact number of unique pending producers, with no
  sentinel.
- `fanin_count == 0` is the only initial-ready criterion.
- No task, heap byte, dependency entry, or TensorMap entry is reclaimed within
  a replay.
