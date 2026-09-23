# Qwen `Worker.submit` production-path evidence

Baseline commit: `ee27950582c1a4c8a92dc9756264cb8ed6ec2f58`
Platform: A3 / CANN 9.0.0 / `host_build_graph` / one local chip

## Delivered

- Fixed workload manifest: `workload_manifest.json`.
- Level-3 `Worker.submit` driver using HOST-backed `Worker.create_buffer`.
- Read-only weights, rope tables, hidden state, and sequence metadata are shared across runs; KV cache and output use per-run buffers.
- One driver owns the run handles, TaskArgs directions, and buffer lifetimes for depth 1 and depth 2.
- Fixture golden validation compares `out`, `k_cache`, and `v_cache`.

## Validation

| Scenario                      | Task                                | Result           | Conclusion                                                                        |
| ----------------------------- | ----------------------------------- | ---------------- | --------------------------------------------------------------------------------- |
| Depth 1 smoke, golden skipped | `task_20260922_214625_200553426315` | exit 0, device 1 | Level-3 `Worker.submit`, A3 HBG, HOST tensor, and single NEXT_LEVEL wiring passed |
| Depth 1 golden                | `task_20260922_214915_21565905113`  | exit 0, device 1 | `out`, 40-layer `k_cache`, and `v_cache` correctness baseline passed              |
| Depth 2 smoke, golden skipped | `task_20260922_215600_246694023753` | exit 1, device 1 | Run 1 completed; run 2 ran out of device memory during bind                       |

The depth-2 device log reported:

```text
rtMalloc failed: 207001 (ACL_ERROR_RT_MEMORY_ALLOCATION)
Retained temp buffer grow failed: required bytes 40860165120
```

Run 2 failed while the runtime grew its retained temporary buffer; run 1 still reached its device boundary and completion. The evidence does not point to a Qwen token/KV dependency, HOST accessor conflict, or P4 identity error.

## Capability matrix and follow-up gap

- Depth 1: supported for the bounded A3 HBG / HOST / single NEXT_LEVEL path.
- Depth 2: the fixed 40-layer Qwen shape exceeds the current resource capacity, so it cannot be admitted under the existing fixed-slot and retained-temp strategy.
- This is a step-three resource, back-pressure, and multi-generation gap. This PR does not change runtime admission, workspace capacity, or reclamation, and it does not present the failed depth-2 run as a serial success.
- A5, TMR, group/SUB, cross-endpoint, DEVICE tensor, dynamic batching, real prefill KV, and full vLLM serving remain outside this evidence.
