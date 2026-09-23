# Qwen3-14B `Worker.submit` production-path driver

This case adapts the existing Qwen3-14B decode callable to the bounded A3 HBG
production path. It uses a level-3 `Worker`, HOST-backed tensors from
`Worker.create_buffer`, and one `Worker.submit` orchestration callback per decode
run. The device still executes one NEXT_LEVEL task; the driver owns the host
buffers and keeps them alive until each run handle is complete.

The workload and environment contract is recorded in
`workload_manifest.json`. The current fixture is a deterministic decode fixture
with synthetic KV state. It is useful for ABI, shape, output, and per-run KV
checks; it is not a claim that a tokenizer prompt or real prefill has been
validated.

## Dependency matrix

| Object | Prepare | Native submit | Required condition |
| --- | --- | --- | --- |
| Weights and rope tables | Early and shared | Yes | Immutable HOST buffers remain alive |
| Hidden state, sequence metadata, block table, slot mapping | Early and shared for this fixture | Yes | Caller keeps the HOST buffers unchanged |
| KV input/output | Per run | Depth 2 only with distinct buffers | Previous producer is complete before reuse |
| Decode output | Per run | Yes | Host reads after the run handle completes |
| Sampled token / request state | Not represented by this fused callable | Unsupported here | A real autoregressive adapter must wait for the preceding result |

The driver supports `--depth 1` and `--depth 2` only. Depth 2 submits both
runs before waiting, then validates each run independently. On the fixed 40-layer
workload, the current runtime reports a retained-temp allocation failure for the
second run because one run needs about 40.86 GiB; this is recorded as a
step-three resource/back-pressure gap. The driver does not infer a performance
gain from host overlap and does not enable A5, TMR, group/SUB, DEVICE tensors,
or dynamic batching.

## Run

```bash
.venv/bin/python examples/a2a3/host_build_graph/qwen3_14b_decode_worker_submit/main.py \
  -p a2a3 -d 0 --depth 1
```

For the bounded early-submit check:

```bash
.venv/bin/python examples/a2a3/host_build_graph/qwen3_14b_decode_worker_submit/main.py \
  -p a2a3 -d 0 --depth 2
```

Evidence for the current main is recorded in `STEP2_REPORT.md`.
