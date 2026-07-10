# simpler-RR replay_graph run summary

This file records the known-good steps used to validate the current
`simpler-RR` workspace. It is meant for another agent to read and rerun
without needing chat history.

## Workspace

- Repo: `/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR`
- Branch during this run: `feat/replay-graph-runtime`
- Commit during this run: `22bc3a72`
- Runtime under test: `replay_graph`
- Platform: `a2a3`
- Use `task-submit --device auto` so the cluster chooses an available card.

## Important environment setup

Always run these commands from `simpler-RR` and inside its venv.

```bash
cd /data/pyptouser/liuchangwen/PTO-RTS/simpler-RR
source .venv/bin/activate
eval "$(pypto-setup --export)"
export PTO_ISA_ROOT=/data/pyptouser/liuchangwen/PTO-RTS/simpler/build/pto-isa
```

Refresh the editable install before running the tests:

```bash
python -m pip install -e . --no-build-isolation
```

Expected result: the editable build finishes with `Successfully installed
simpler-0.1.0`.

Do not run the editable install from the outer shell without activating the
venv. That failed with `ModuleNotFoundError: No module named
'scikit_build_core'` because it used the wrong Python environment.

## Test 1: three-loop paged attention

Case file:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/examples/a2a3/replay_graph/batch_paged_attention_three_loop/test_batch_paged_attention_three_loop.py
```

Run command. Keep this as a single line; multiline quoting broke the generated
`task-submit` script with `unexpected EOF while looking for matching quote`.

```bash
task-submit --device auto --max-time 900 --timeout 1200 --run "bash -lc 'cd /data/pyptouser/liuchangwen/PTO-RTS/simpler-RR && source .venv/bin/activate && eval \"\$(pypto-setup --export)\" && export PTO_ISA_ROOT=/data/pyptouser/liuchangwen/PTO-RTS/simpler/build/pto-isa && python examples/a2a3/replay_graph/batch_paged_attention_three_loop/test_batch_paged_attention_three_loop.py -p a2a3 -d \"\$TASK_DEVICE\" --runtime replay_graph --enable-l2-swimlane 4 --log-level v5'"
```

Known-good result from 2026-07-09:

```text
TestBatchPagedAttentionThreeLoop::CaseSmall1ThreeLoop ... PASSED
exit=0
device=2
task_id=task_20260709_203412_65653521097
```

Log:

```text
/data/pyptouser/taskqueue/logs/task_20260709_203412_65653521097.log
```

Latest output/profiling directory from that run:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestBatchPagedAttentionThreeLoop_CaseSmall1ThreeLoop_20260709_203430
```

Profiling files:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestBatchPagedAttentionThreeLoop_CaseSmall1ThreeLoop_20260709_203430/merged_swimlane.json
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestBatchPagedAttentionThreeLoop_CaseSmall1ThreeLoop_20260709_203430/l2_swimlane_records.json
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestBatchPagedAttentionThreeLoop_CaseSmall1ThreeLoop_20260709_203430/name_map_TestBatchPagedAttentionThreeLoop_CaseSmall1ThreeLoop.json
```

## Test 2: explicit qwen 3-layer replay_graph case

Case file:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/examples/a2a3/replay_graph/qwen3_14b_decode_3layer/test_qwen3_14b_decode.py
```

This is the explicit 3-layer replay_graph validation case in `simpler-RR`.
Do not replace it with the pto-lib fused qwen command when validating
per-layer replay_graph behavior.

Run command:

```bash
task-submit --device auto --max-time 1800 --timeout 2400 --run "bash -lc 'cd /data/pyptouser/liuchangwen/PTO-RTS/simpler-RR && source .venv/bin/activate && eval \"\$(pypto-setup --export)\" && export PTO_ISA_ROOT=/data/pyptouser/liuchangwen/PTO-RTS/simpler/build/pto-isa && python examples/a2a3/replay_graph/qwen3_14b_decode_3layer/test_qwen3_14b_decode.py -p a2a3 -d \"\$TASK_DEVICE\" --runtime replay_graph --enable-l2-swimlane 4 --log-level v5'"
```

Known-good result from 2026-07-09:

```text
TestQwen314BDecode3Layer::StressBatch16Seq3500 ... PASSED
exit=0
device=4
task_id=task_20260709_203449_67283731399
```

Log:

```text
/data/pyptouser/taskqueue/logs/task_20260709_203449_67283731399.log
```

Latest output/profiling directory from that run:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestQwen314BDecode3Layer_StressBatch16Seq3500_20260709_203546
```

Profiling files:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestQwen314BDecode3Layer_StressBatch16Seq3500_20260709_203546/merged_swimlane.json
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestQwen314BDecode3Layer_StressBatch16Seq3500_20260709_203546/l2_swimlane_records.json
/data/pyptouser/liuchangwen/PTO-RTS/simpler-RR/outputs/TestQwen314BDecode3Layer_StressBatch16Seq3500_20260709_203546/name_map_TestQwen314BDecode3Layer_StressBatch16Seq3500.json
```

## Expected warnings

The following warnings were seen and did not indicate failure:

```text
Warning: The /usr/local/Ascend/cann-9.0.0 owner does not match the current owner.
Permission mismatch: torch_npu/lib/libop_plugin_atb.so owner does not match.
```

Continue waiting for the final `PASSED` or failure line.

## Troubleshooting notes

- If a `task-submit` run is disconnected, reconnect with:

```bash
task-submit --wait <task_id>
```

- If a device fails to initialize, retry with `--device auto`. Device allocation
  or `rtSetDevice` failures are usually cluster/environment issues, not proof of
  a code failure.
- Keep the `task-submit --run` payload as one shell line unless you regenerate
  the quoting very carefully.
- For these two `simpler-RR` tests, no extra pto-lib `PYTHONPATH` override is
  needed. The extra pto-lib path setup was only needed for running the separate
  pto-lib qwen script.
- Use this `PTO_ISA_ROOT` for the current successful setup:

```text
/data/pyptouser/liuchangwen/PTO-RTS/simpler/build/pto-isa
```
