"""Make a compact, prompt-free result bundle from retained replay evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = {"arms": {}}
    for arm in ("baseline", "exact", "exact-recovery", "fast"):
        root = a.raw / arm
        if not (root / "summary.json").is_file():
            continue
        summary = json.loads((root / "summary.json").read_text())
        summary["requests"] = []
        for path in sorted(root.glob("*.json")):
            if path.name == "summary.json":
                continue
            row = json.loads(path.read_text())
            summary["requests"].append({k: row.get(k) for k in (
                "name", "started_unix", "seconds", "ttft_seconds", "tpot_seconds", "chunk_gap_p50_seconds",
                "chunk_gap_p95_seconds", "usage", "finish_reason", "done", "passed", "error")})
            summary["requests"][-1].update(
                content_sha256=hashlib.sha256(row["text"].encode()).hexdigest(),
                messages_sha256=hashlib.sha256(json.dumps(row["request"]["messages"], sort_keys=True).encode()).hexdigest())
        campaign_rows = [r for r in summary["requests"] if r["name"].startswith("campaign")]
        summary["campaign_distinct_response_texts"] = len({r["content_sha256"] for r in campaign_rows})
        summary["campaign_distinct_message_payloads"] = len({r["messages_sha256"] for r in campaign_rows})
        warm = [r for r in summary["requests"] if r["name"].startswith("session") and not r["name"].endswith("turn-0")]
        if warm:
            quantile = lambda key, q: sorted(r[key] for r in warm)[math.ceil(q * len(warm)) - 1]
            summary["session_warm_turns"] = {"n": len(warm),
                "ttft_p50_seconds": quantile("ttft_seconds", .5), "ttft_p95_seconds": quantile("ttft_seconds", .95),
                "tpot_p50_seconds": quantile("tpot_seconds", .5), "tpot_p95_seconds": quantile("tpot_seconds", .95)}
        if arm == "baseline":
            def counter(path, name):
                return sum(float(line.rsplit(" ", 1)[1]) for line in path.read_text().splitlines()
                           if line.startswith("vllm:" + name + "{"))
            summary["counter_deltas"] = {name: counter(root / "after-metrics.txt", name) - counter(root / "before-metrics.txt", name)
                for name in ("request_success_total", "prompt_tokens_total", "generation_tokens_total", "prefix_cache_hits_total", "prefix_cache_queries_total")}
            summary["generation_counters_match_replay"] = (
                summary["counter_deltas"]["request_success_total"] == len(summary["requests"]) and
                summary["counter_deltas"]["prompt_tokens_total"] == sum(r["usage"].get("prompt_tokens", 0) for r in summary["requests"]) and
                summary["counter_deltas"]["generation_tokens_total"] == sum(r["usage"].get("completion_tokens", 0) for r in summary["requests"]))
        summary["engine_snapshots"] = {}
        for path in sorted(root.glob("*-get_server_info.txt")):
            try:
                snapshot = json.loads(path.read_text())
                summary["engine_snapshots"][path.stem] = {key: snapshot.get(key) for key in (
                    "config", "memory_gib", "ledger_gib", "scheduler", "spec_stats", "w3_target_identity", "w3_warning")}
            except ValueError:
                pass
        memory_path = a.raw / ("spark1-memory.jsonl" if arm == "baseline" else "spark2-memory.jsonl")
        if memory_path.is_file():
            samples = [json.loads(line) for line in memory_path.read_text().splitlines() if line.startswith("{")]
            samples = [s for s in samples if summary["started_unix"] <= s["unix"] <= summary["finished_unix"]]
            if samples:
                summary["global_memory"] = {"samples": len(samples),
                    "min_available_gib": min(s["MemAvailable_kib"] for s in samples) / 2**20,
                    "start_available_gib": samples[0]["MemAvailable_kib"] / 2**20,
                    "end_available_gib": samples[-1]["MemAvailable_kib"] / 2**20,
                    "swap_used_start_gib": (samples[0]["SwapTotal_kib"] - samples[0]["SwapFree_kib"]) / 2**20,
                    "swap_used_end_gib": (samples[-1]["SwapTotal_kib"] - samples[-1]["SwapFree_kib"]) / 2**20}
        result["arms"][arm] = summary
    baseline = result["arms"].get("baseline", {}).get("groups", {})
    for arm in ("exact", "fast"):
        candidate = result["arms"].get(arm)
        if not candidate:
            continue
        candidate["ratios_to_baseline"] = {}
        for group in ("campaign", "session", "structured"):
            if group not in baseline or group not in candidate["groups"]:
                continue
            b, c = baseline[group], candidate["groups"][group]
            candidate["ratios_to_baseline"][group] = {
                "validated_request_throughput": c["validated_requests_per_hour"] / b["validated_requests_per_hour"] if b["validated_requests_per_hour"] else None,
                "output_token_wall_throughput": c["output_tokens_per_wall_second"] / b["output_tokens_per_wall_second"] if b["output_tokens_per_wall_second"] else None,
                "baseline_passed": b["passed"], "candidate_passed": c["passed"]}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({arm: {"passed": data["passed"], "groups": data["groups"], "ratios": data.get("ratios_to_baseline"), "memory": data.get("global_memory")} for arm, data in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
