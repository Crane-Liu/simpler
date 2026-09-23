# Qwen step-two workload contract

Status: source audit complete; the concrete external fixture and checkpoint artifact are still required before the contract can be frozen for execution.

## Existing source of truth

The repository already contains a post-prefill Qwen3-14B HBG serving path under `examples/a2a3/host_build_graph/qwen3_14b_serving_effective/`. Its checked-in reference manifest records:

| Field | Value |
| --- | --- |
| Model | Qwen3-14B |
| Dtype | BF16 |
| Prompt tokens | 3338 |
| Batch size | 16 |
| Output tokens | 128 |
| Decode dispatches | 127 consumed dispatches |
| Steady skip | 5 dispatches |
| Runtime slots | 0 and 1 |
| KV backing pages | 691 |
| Runtime | A3 `host_build_graph` |
| Definition shape | One Definition invoked 40 times per frame |
| ABI | 25 arguments, or 26 with `sampled_ids_host` |

This reference is a workload shape contract. It does not contain the external model checkpoint, prefill snapshot, generated decode artifact, tokenizer files, or their checksums.

## Prefill snapshot contract

`fixture.py` defines the executable fixture schema as `serving-tmr-standalone-fixture-v1`. The fixture must prove all of the following before decode starts:

- all chunked prefill work is complete;
- the first token has been produced;
- no decode dispatch has run before the snapshot;
- prefill and sampling fences have retired;
- prompt token IDs have shape `[16, 3338]` and dtype `int32`;
- the first generated token, sequence lengths, block table, next slot mapping, and token counts are mutually consistent;
- every KV shard and metadata file matches its SHA256 manifest.

The first decode input is the prefill fixture's `first_generated_token_ids`. The first decode position is named by `next_slot_mapping`; subsequent positions advance the page/block metadata in the golden contract. The fixture provides golden sampled-token rows for the remaining decode dispatches.

## Runtime and artifact contract

The real serving driver validates a caller-provided Qwen checkpoint, fixture, and generated decode artifact. The artifact builder accepts only the 25-argument compatibility ABI or the 26-argument ABI that adds `sampled_ids_host`. The required semantic arguments include `out`, `embed_weight`, `sampled_ids_in`, `sampled_ids`, and `next_hidden`; the exact order comes from the artifact's `distributed_meta.json`.

The generated artifact must preserve the source orchestration shared library and every in-core binary by checksum. Its Definition must contain fewer than 1024 tasks per layer and invoke one recorded layer Definition 40 times per frame.

## Fields that remain unfrozen

The repository does not contain the external fixture root or model checkpoint. The following values must be supplied and recorded before the real workload is declared frozen:

- prompt token-ID SHA256;
- tokenizer and model checkpoint identity plus file SHA256 values;
- CANN, torch-npu, vLLM/vLLM-Ascend, PyPTO, PyPTO-lib, and PTOAS versions;
- fixture manifest SHA256, metadata SHA256, KV shard checksums, and golden output-token SHA256;
- generated artifact manifest, `distributed_meta.json`, orchestration shared-library, and in-core binary checksums;
- the exact `sampled_ids_host` ABI selected for the Worker adapter.

The current `qwen3_14b_decode_worker_submit` manifest is therefore an implementation probe. It must be replaced or extended with these concrete values before step two closes.
