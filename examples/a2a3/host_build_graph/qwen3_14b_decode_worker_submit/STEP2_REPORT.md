# Qwen `Worker.submit` production-path evidence

基线提交：`ee27950582c1a4c8a92dc9756264cb8ed6ec2f58`
平台：A3 / CANN 9.0.0 / `host_build_graph` / 单本地 chip
工作树：`codex/vllm-step2-qwen`

## 已交付

- 固定 workload manifest：`workload_manifest.json`。
- 新增 level-3 `Worker.submit` driver，使用 HOST-backed `Worker.create_buffer`。
- 只读权重、rope、hidden 和 sequence metadata 在多个 run 间共享；KV cache 与 output 按 run 使用独立 buffer。
- depth=1 和 depth=2 的 run handle、TaskArgs 方向和资源生命周期由同一 driver 组织。
- Pure/fixture golden 路径比较 `out`、`k_cache` 和 `v_cache`。

## 验证结果

| 场景 | task | 结果 | 结论 |
| --- | --- | --- | --- |
| depth=1 smoke，跳过 golden | `task_20260922_214625_200553426315` | exit 0，设备 1 | L3 `Worker.submit`、A3 HBG、HOST tensor、单 NEXT_LEVEL 接线通过 |
| depth=1 golden | `task_20260922_214915_21565905113` | exit 0，设备 1 | `out`、40 层 `k_cache`、`v_cache` 正确性基线通过 |
| depth=2 smoke，跳过 golden | `task_20260922_215600_246694023753` | exit 1，设备 1 | run1 完成；run2 bind 时设备内存不足 |

depth=2 的设备日志报告：

```text
rtMalloc failed: 207001 (ACL_ERROR_RT_MEMORY_ALLOCATION)
Retained temp buffer grow failed: required bytes 40860165120
```

run2 的失败发生在 runtime bind 的 retained temporary buffer 扩容阶段；run1 的设备边界和 completion 仍然有效。当前证据不支持把它归因为 Qwen token/KV 数据依赖、HOST accessor 冲突或 P4 身份错误。

## 能力矩阵与后续缺口

- depth=1：当前 bounded A3 HBG / HOST / 单 NEXT_LEVEL 路径支持。
- depth=2：当前 40 层 Qwen shape 受资源容量限制，不能在现有固定两槽和 retained-temp 策略下放行。
- 该缺口属于步骤三的资源/背压/多代在途工作；本 PR 不修改 runtime admission、workspace 容量或回收策略，也不把失败伪装成串行成功。
- A5、TMR、group/SUB、跨 endpoint、DEVICE tensor、动态 batch、真实 prefill KV 和完整 vLLM serving 仍未覆盖。
