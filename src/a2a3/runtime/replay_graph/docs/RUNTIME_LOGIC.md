# Replay Graph Runtime Logic

## 1. Execution Contract

`replay_graph` pipelines device orchestration and scheduling at explicit graph
boundaries:

1. One AICPU orchestrator builds graph N in the active graph arena.
2. `rt_graph_boundary()` publishes graph N and releases its task publish gates.
3. Scheduler threads execute graph N while the orchestrator builds graph N+1
   in the other arena.
4. The orchestrator reuses an arena only after its tasks have completed and the
   following graph has finished adding cross-graph dependency edges.

The final `rt_orchestration_done()` call publishes any remaining tasks and
marks the task stream complete. Scheduler threads start polling immediately
after runtime initialization; there is no whole-orchestration dispatch barrier.

## 2. Lifecycle

Task state is monotonic:

```text
PENDING -> COMPLETED
```

There is no `CONSUMED` state. Graph-level arena controls provide the reuse
contract instead:

```text
FREE -> BUILDING -> RUNNING -> DONE -> FREE
```

`exec_done` means every task in the graph completed. `dep_closed` means the
following graph can no longer append consumers to this graph's producers. Both
conditions must hold before the physical task and heap arena is reused.

Scopes preserve nesting, manual-dependency, DFX, and profiling semantics. A
scope does not publish a graph and does not own an arena; only
`rt_graph_boundary()` changes graph ownership.

## 3. Memory Ownership

### Shared memory

`PTO2SharedMemoryHeader` contains:

- the monotonic logical task count;
- `task_descriptors[]`, `task_payloads[]`, and `slot_states[]`;
- `task_slot_map[]`, which maps a logical task id to a physical arena slot;
- final orchestration, output, fatal-error, and stall state.

Logical task ids remain dense and monotonic. Physical slots come from one of
two half-window arenas, so task id and slot are not generally identical.

### Orchestrator arena

The orchestrator owns:

- `PTO2TaskAllocator`, with two task-slot and heap bump arenas;
- `PTO2DepListPool`, which holds fanout edges for the full invocation;
- `fanin_seen_epoch[]`, used for per-submit producer deduplication;
- `PTO2TensorMap`, used for automatic tensor dependency discovery;
- graph-cache templates and record state.

The dependency pool and TensorMap remain monotonic for the invocation. Only
task slots and output heap bytes use graph-level ping-pong reuse.

### Scheduler arena

The scheduler owns ready queues, early-dispatch queues, async wait state, and
profiling counters. It updates graph completion counters after the completed
task's fanout walk is finished, so an arena cannot be reused while a scheduler
still reads one of its slots.

## 4. Allocation

`PTO2TaskAllocator` divides the configured task window and heap equally between
two graph arenas. Each successful allocation:

1. reserves the next logical task id;
2. reserves the next physical slot in the active arena;
3. writes the logical-id-to-slot mapping;
4. aligns and reserves output storage in the active heap arena;
5. release-publishes the new logical task count.

One graph must fit in one half of the task window and heap. Dependency-pool and
TensorMap capacities must hold the complete orchestration invocation.

## 5. Submit And Concurrent Wiring

Every non-inline task starts with one synthetic publish dependency:

```text
fanin_count = 1
fanin_refcount = 0
```

`submit_task` then discovers explicit and TensorMap dependencies. For each
unique pending producer, it:

1. increments the consumer's `fanin_count` before publishing the edge;
2. pushes the consumer onto the producer's atomic `fanout_head` stack;
3. treats the dependency as already satisfied if completion closed the stack
   before the compare-and-swap succeeded.

Publishing the count before the edge prevents a concurrent producer completion
from making an incompletely built consumer ready. A producer completion
atomically exchanges `fanout_head` with a CLOSED sentinel. An orchestrator
append therefore either joins the list before closure or observes completion
and omits the runtime edge.

Graph recording captures the logical dependency before the completion check.
The cached topology therefore does not depend on whether a prior-graph producer
happened to finish while the next graph was being recorded.

## 6. Graph Boundary

`rt_graph_boundary()` performs the publication protocol:

1. seals the active graph's logical task range;
2. counts inline-completed allocation tasks;
3. publishes the range and graph completion state;
4. releases each pending task's synthetic publish dependency;
5. closes dependency wiring for the graph in the next arena;
6. waits until that arena has both `exec_done` and `dep_closed`;
7. resets the reusable task and heap bumps and starts building there.

The boundary release is the graph-freeze point. A task can receive completed
producer notifications while it is being built, but it cannot enter a ready
queue until its publish dependency is released.

## 7. Scheduling And Completion

Scheduler threads poll ready queues while orchestration is active. For a normal
completion, the scheduler:

1. marks the producer `COMPLETED`;
2. atomically closes and snapshots its fanout stack;
3. increments each consumer's `fanin_refcount`;
4. routes a consumer exactly once when
   `fanin_refcount == fanin_count`;
5. updates the producer graph's completion count after the fanout walk.

Dummy and deferred-completion tasks use the same final completion hook, so they
participate in arena reuse accounting without special cases.

The scheduler exits only after final orchestration completion and the total
completed task count reaches the final logical task count.

## 8. Graph Record And Replay

`PTO2_GRAPH_SCOPE(key, bindings)` records a task-DAG template on a cache miss.
On a hit, the runtime skips the orchestration block and materializes task
descriptors, payload values, outputs, and fanout edges directly in the active
graph arena.

Replayed tasks receive the same synthetic publish dependency as normally
submitted tasks. Replay does not seed a separate initial-ready list;
`rt_graph_boundary()` releases the restored DAG after all slots and edges are
materialized.

The V1 cache is process-local and supports boundary-tensor write graphs. Cache
keys include the runtime schema, callable content hash, user namespace, tensor
metadata, and bound scalar values.

## 9. Early Dispatch

The publish dependency also participates in `dispatch_fanin`. Boundary
publication accounts for that dependency before releasing completion fanin.
Internal producer launches can then advance `dispatch_fanin` to the task's
final `fanin_count` and retain the existing early-dispatch behavior.

A cross-graph producer may publish before or during the boundary operation. In
that race, early staging is opportunistic; completion readiness remains exact
because it uses the atomic fanout-close protocol and `fanin_refcount`.

## 10. TensorMap And Data Access

TensorMap entries remain valid for the invocation and are reset by epoch between
runtime invocations. Newer producer entries shadow prior graph entries for the
same tensor region.

An entry stores a logical producer task id. Resolving that id through
`task_slot_map` is valid only while the slot's task descriptor still carries the
same logical id. A mismatch means the producer's arena has already completed
and been reused, so the dependency is already satisfied and must be skipped.
This snapshot check prevents an old TensorMap entry or cached external fanin
from being attached to the unrelated task that now occupies the physical slot.

`get_tensor_data` and `set_tensor_data` can wait for producer completion, but
there is no consumer-retirement state. `set_tensor_data` therefore provides WAW
protection, not WAR protection.

## 11. Capacity Failures

The following fail fast:

- one graph exceeds half of the task window;
- one graph exceeds half of the output heap;
- the invocation exceeds the dependency pool;
- the invocation exceeds TensorMap capacity;
- a graph-cache template exceeds its fixed record/export capacities.

Graph-cache capacity failure falls back to normal orchestration when no runtime
state has been partially materialized. Runtime arena exhaustion is fatal.

## 12. Invariants

- `ring_id == 0` for every task.
- Logical task ids are dense; `task_slot_map` resolves physical slots.
- A task has one publish dependency plus its unique pending producers.
- A graph is dispatchable only after `rt_graph_boundary()` releases its publish
  dependencies.
- Fanout append and completion-close are atomic and cannot lose an edge.
- A resolved dependency slot is used only when its task-id snapshot matches the
  requested logical producer id.
- A physical arena is reused only when `exec_done && dep_closed`.
- Dependency-pool and TensorMap storage are not reclaimed within an invocation.
