# Qwen step-two execution and lifetime map

Status: source audit complete. This map describes the existing real serving path and the adapter boundary needed to move it to Simpler `Worker.submit`.

## Existing real serving path

```text
prefill fixture
  -> metadata + KV shard validation
  -> model checkpoint and generated artifact validation
  -> DistributedCompiledProgram.prepare
  -> resident weights / RoPE / KV / output / sampled buffers
  -> one decode submit per token step
  -> handle.result
  -> sampled token readback
  -> next-step metadata update and decode submit
```

The single runner waits for every step before submitting the next step. The dual runner submits the first two steps before waiting, then keeps a two-slot pending queue. It alternates slots, reads sampled IDs before a slot is reused, and validates that device runner spans remain serialized.

## Per-step argument map

| Object | Producer | Consumer | Lifetime |
| --- | --- | --- | --- |
| Weights | checkpoint loader | every layer | all frames |
| RoPE tables | checkpoint/fixture loader | every decode step | all frames |
| KV cache | prefill fixture and decode kernels | attention in later steps | device resident; shared by ordered steps |
| Block table | fixture metadata plus host slot update | attention/page lookup | until the submitted step consumes it |
| Slot mapping | fixture metadata plus host slot update | attention KV write position | until the submitted step consumes it |
| `sampled_ids_in` | initial fixture token or previous decode output | next decode step | device-only producer/consumer chain |
| `sampled_ids` | decode sampler | next decode step and host validator | until the next consumer and host read finish |
| `sampled_ids_host` | device copyback ABI, when present | host validator/request layer | until host read completes |
| `next_hidden` | decoder | next decode step | per in-flight slot |
| `out` | LM head | host validator/request layer | per in-flight slot until readback |

## Dependency classification

- Weights and RoPE are immutable and can be prepared early and shared.
- KV cache is mutable device state. A later step may read and update it only after the device has completed the earlier producer in the same ordered execution domain.
- `sampled_ids_in` can remain a device-only dependency when the next orchestration only passes its address and the device queue preserves the producer-before-consumer order. Host preparation must not read its contents.
- `sampled_ids_host` is a host-visible result. A slot cannot be recycled until the host has consumed it and the corresponding run handle has completed.
- Block table, slot mapping, and sequence lengths are host metadata. The checked-in fixture precomputes their per-step values. A real vLLM scheduler must produce the next values before the corresponding host prepare; the runtime must not infer them from a future device result.
- `next_hidden` and `out` need independent storage for two in-flight slots. Sharing either buffer would overwrite the predecessor result.

## Current Worker.submit adapter boundary

The existing step-two probe already demonstrates the outer path:

```text
Worker(level=3)
  -> register ChipCallable
  -> init
  -> Worker.submit(orch_fn)
  -> orch.submit_next_level(chip_handle, HOST TaskArgs, config, worker=0)
  -> RunHandle.wait/result
```

Its callable is the checked-in synthetic 20-argument Qwen fixture. The real serving artifact is different: it is an external generated `DistributedCompiledProgram` with a 25/26-argument ABI, a generated orchestration shared library, and precompiled in-core binaries. `compile_chip_callable_spec` cannot consume that directory directly. The next implementation must choose one explicit bridge:

1. build a `ChipCallable` from the artifact's orchestration and in-core binaries while preserving the artifact ABI; or
2. regenerate the same callable through the repository's source-based `ChipCallable` path and prove byte/ABI equivalence to the serving artifact.

The adapter must keep the external artifact and fixture checksums in its manifest. It must also express the `sampled_ids_host` result as a HOST output buffer when the 26-argument ABI is selected.

## Current conclusions

- The real post-prefill fixture already supplies the token, KV, block-table, slot-mapping, and golden-output contracts needed for step 2.
- The current synthetic `Worker.submit` probe is useful for HOST buffer and run-lifetime validation but cannot close real Qwen correctness.
- The first implementation task after review is the artifact/fixture bridge and a single real decode step through `Worker.submit`. Runtime admission and resource policy remain outside this phase.
