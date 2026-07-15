#!/usr/bin/env python3
"""Build a compact Host-O / device-S Perfetto trace from one profiled run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STRACE_RE = re.compile(
    r"\[STRACE\].*?\binv=(?P<inv>\d+).*?\bname=(?P<name>\S+) "
    r"ts=(?P<ts>\d+) dur=(?P<dur>\d+)(?: (?P<attrs>.*))?$"
)


def parse_attrs(raw: str | None) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for token in (raw or "").split():
        if "=" in token:
            key, value = token.split("=", 1)
            attrs[key] = value
    return attrs


def load_host_spans(log_path: Path) -> dict[int, list[dict[str, object]]]:
    rounds: dict[int, list[dict[str, object]]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STRACE_RE.search(line)
        if match is None or not match.group("name").startswith("simpler_run.host_orch."):
            continue
        invocation = int(match.group("inv"))
        rounds.setdefault(invocation, []).append(
            {
                "name": match.group("name"),
                "ts_ns": int(match.group("ts")),
                "dur_ns": int(match.group("dur")),
                "attrs": parse_attrs(match.group("attrs")),
            }
        )
    if not rounds:
        raise ValueError(f"no Host-O STRACE spans found in {log_path}")
    return rounds


def layer_value(span: dict[str, object]) -> int:
    return int(span["attrs"].get("layer", "-1"))  # type: ignore[union-attr]


def find_span(spans: list[dict[str, object]], name: str, layer: int) -> dict[str, object]:
    matches = [span for span in spans if span["name"] == name and layer_value(span) == layer]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} span for layer {layer}, found {len(matches)}")
    return matches[0]


def find_optional_span(
    spans: list[dict[str, object]], name: str, layer: int,
) -> dict[str, object] | None:
    matches = [span for span in spans if span["name"] == name and layer_value(span) == layer]
    if len(matches) > 1:
        raise ValueError(f"expected at most one {name} span for layer {layer}, found {len(matches)}")
    return matches[0] if matches else None


def metadata_event(pid: int, tid: int, kind: str, value: str) -> dict[str, object]:
    return {"ph": "M", "name": kind, "pid": pid, "tid": tid, "args": {"name": value}}


def complete_event(
    pid: int, tid: int, name: str, category: str, ts_ns: int, dur_ns: int, base_ns: int,
    args: dict[str, object],
) -> dict[str, object]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "name": name,
        "cat": category,
        "ts": (ts_ns - base_ns) / 1000.0,
        "dur": dur_ns / 1000.0,
        "args": args,
    }


def build_round_events(
    invocation: int, round_index: int, spans: list[dict[str, object]], l2_path: Path,
) -> list[dict[str, object]]:
    data = json.loads(l2_path.read_text(encoding="utf-8"))
    frequency = int(data["metadata"]["clock_freq_hz"])
    aicore_tasks = data["aicore_tasks"]
    task_layers = sorted(
        {
            layer_value(span)
            for span in spans
            if span["name"] == "simpler_run.host_orch.layer_build" and int(span["attrs"].get("tasks", "0")) > 0
        }
    )
    if not task_layers:
        raise ValueError(f"invocation {invocation} has no task-bearing Host-O layers")

    task_begin = 0
    layer_ranges: dict[int, tuple[int, int]] = {}
    for layer in task_layers:
        build = find_span(spans, "simpler_run.host_orch.layer_build", layer)
        task_count = int(build["attrs"]["tasks"])
        layer_ranges[layer] = (task_begin, task_begin + task_count)
        task_begin += task_count

    s_layers: list[dict[str, object]] = []
    for layer in task_layers:
        begin, end = layer_ranges[layer]
        records = [record for record in aicore_tasks if begin <= int(record[1]) < end]
        if not records:
            raise ValueError(f"{l2_path} has no AICore records for task range [{begin},{end})")
        start_cycles = min(int(record[3]) for record in records)
        end_cycles = max(int(record[4]) for record in records)
        duration_ns = round((end_cycles - start_cycles) * 1_000_000_000 / frequency)
        completion_wait = find_span(spans, "simpler_run.host_orch.boundary.wait", layer + 1)
        ack_ns = int(completion_wait["ts_ns"]) + int(completion_wait["dur_ns"])
        s_layers.append(
            {
                "layer": layer,
                "start_ns": ack_ns - duration_ns,
                "dur_ns": duration_ns,
                "records": len(records),
                "unique_tasks": len({int(record[1]) for record in records}),
            }
        )

    base_ns = min(
        min(int(span["ts_ns"]) for span in spans), min(int(layer["start_ns"]) for layer in s_layers)
    )
    pid = round_index + 1
    events: list[dict[str, object]] = [
        metadata_event(pid, 0, "process_name", f"Qwen Host O / Device S - round {round_index}"),
        metadata_event(pid, 1, "thread_name", "O active (Host build/materialize/stage/commit)"),
        metadata_event(pid, 2, "thread_name", "O synchronization (S release/completion)"),
        metadata_event(pid, 3, "thread_name", "S execution (AICore envelope)"),
    ]

    phase_names = {
        "simpler_run.host_orch.layer_build": "build",
        "simpler_run.host_orch.boundary.materialize": "materialize",
        "simpler_run.host_orch.boundary.stage_upload": "stage upload",
        "simpler_run.host_orch.boundary.commit": "commit",
        "simpler_run.host_orch.boundary.upload": "upload",
    }
    for span in spans:
        phase = phase_names.get(str(span["name"]))
        if phase is None or layer_value(span) not in task_layers:
            continue
        layer = layer_value(span)
        attrs = dict(span["attrs"])
        cache_mode = attrs.get("cache", "none")
        events.append(
            complete_event(
                pid, 1, f"O L{layer} {phase} ({cache_mode})", "host_orchestrator",
                int(span["ts_ns"]), int(span["dur_ns"]), base_ns,
                {"layer": layer, "phase": phase, **attrs},
            )
        )

    for layer in task_layers[1:] + [task_layers[-1] + 1]:
        wait = find_span(spans, "simpler_run.host_orch.boundary.wait", layer)
        completed_layer = layer - 1
        events.append(
            complete_event(
                pid, 2, f"wait for S L{completed_layer}", "host_backpressure",
                int(wait["ts_ns"]), int(wait["dur_ns"]), base_ns,
                {"completed_layer": completed_layer, "timing": "device completion polling"},
            )
        )

    for layer in task_layers:
        release_wait = find_optional_span(
            spans, "simpler_run.host_orch.boundary.release_wait", layer
        )
        if release_wait is None:
            continue
        events.append(
            complete_event(
                pid, 2, f"wait for S L{layer} release", "host_release_handshake",
                int(release_wait["ts_ns"]), int(release_wait["dur_ns"]), base_ns,
                {"layer": layer, "timing": "device graph release polling"},
            )
        )

    for s_layer in s_layers:
        layer = int(s_layer["layer"])
        begin, end = layer_ranges[layer]
        events.append(
            complete_event(
                pid, 3, f"S L{layer} ({s_layer['records']} AICore records)", "device_scheduler",
                int(s_layer["start_ns"]), int(s_layer["dur_ns"]), base_ns,
                {
                    "layer": layer,
                    "task_range": f"[{begin},{end})",
                    "records": s_layer["records"],
                    "unique_task_ids": s_layer["unique_tasks"],
                    "timing": "actual AICore duration; end aligned to Host-O completion acknowledgement",
                },
            )
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="run.log containing Host-O STRACE spans")
    parser.add_argument("--l2-dir", type=Path, required=True, help="test output containing round_NNN L2 JSON files")
    parser.add_argument("--output", type=Path, required=True, help="output Perfetto JSON")
    args = parser.parse_args()

    rounds = load_host_spans(args.log)
    events: list[dict[str, object]] = []
    for round_index, invocation in enumerate(sorted(rounds)):
        l2_path = args.l2_dir / f"round_{round_index:03d}" / "l2_swimlane_records.json"
        if not l2_path.is_file():
            raise FileNotFoundError(l2_path)
        events.extend(build_round_events(invocation, round_index, rounds[invocation], l2_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"traceEvents": events}, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
