"""Failure-driven localization probe: does Laya rank the buggy function better
than the evidence already in the failing test output?

Runs the config grid over a built corpus (see `build_failure_loc.py`), computes
the deterministic baselines on the same shortlist, and applies the go/no-go
gates from `docs/failure-localization-plan.md`.

    python scripts/localize_failure.py --root /data/laya-ft/failure-loc \
      --configs L,L0,L1 --k 12 --tail 2000 --baselines --out results.json

Every model call is cached in `<root>/deepseek-cache.jsonl` keyed by
(instance, config, k, tail, model), so re-analysis costs nothing.

Configs:
    L   task + failure output      (the feature)
    L0  task only                  (today's issue-text probe)
    L1  failure output only

Baselines (no model, same shortlist):
    H1  innermost source traceback frame
    H2  BM25(failure + task) over candidate text
    H3  the deterministic shortlist order itself (frame-weighted fusion)
    RND analytic 1/k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import localize_candidates as lc  # noqa: E402


# --- corpus -----------------------------------------------------------------

def load_instance(root: str, iid: str) -> dict | None:
    base = os.path.join(root, iid)
    status = lc_path = None
    try:
        with open(os.path.join(base, "status.json")) as handle:
            status = json.load(handle)
        with open(os.path.join(base, "candidates.json")) as handle:
            candidates = json.load(handle)["candidates"]
        with open(os.path.join(base, "gold.json")) as handle:
            gold = json.load(handle)
        with open(os.path.join(base, "failure.txt")) as handle:
            failure = handle.read()
        with open(os.path.join(base, "task.json")) as handle:
            task = json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    if not status or not status.get("ok"):
        return None
    problem = (task.get("problem_statement") or "").strip()
    if not problem:
        # No issue text in this SWE-smith row. Use the failing node ids, never the
        # assertion detail: otherwise the "task only" ablation would leak the
        # failure output it is supposed to isolate.
        failed = lc.parse_failed_tests(failure)[:5]
        problem = "Fix the failing tests: " + ", ".join(failed) if failed else "Fix the failing tests."
    return {
        "instance_id": iid,
        "root": base,
        "repo": os.path.join(base, "repo"),
        "candidates": candidates,
        "gold": gold,
        "failure": failure,
        "problem": problem,
    }


def discover(root: str) -> list[dict]:
    instances = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        item = load_instance(root, name)
        if item is not None:
            instances.append(item)
    return instances


# --- ranking ----------------------------------------------------------------

def frame_hits(candidate: dict, frames: list[dict]) -> int:
    if candidate["kind"] != "function":
        return 0
    return sum(
        1 for frame in frames
        if frame["file"] == candidate["path"] and candidate["line"] <= frame["line"] <= candidate["end"]
    )


def order_h1(shortlist: list[dict], frames: list[dict], failure: str, task: str) -> list[str]:
    """Frame-first order; falls back to the shortlist order when no frame hits."""
    source = [frame for frame in frames if not lc.is_test_path(frame["file"])]
    if not source:
        return [candidate["id"] for candidate in shortlist]
    hit = [candidate for candidate in shortlist if frame_hits(candidate, source) > 0]
    rest = [candidate for candidate in shortlist if frame_hits(candidate, source) == 0]
    return [candidate["id"] for candidate in hit + rest]


def order_h2(shortlist: list[dict], failure: str, task: str) -> list[str]:
    query = lc.tokens(lc.query_text(failure)) + lc.tokens(task)
    docs = [lc.tokens(candidate["text"]) for candidate in shortlist]
    scores = lc.bm25(query, docs)
    rows = sorted(zip(shortlist, scores), key=lambda row: -row[1])
    return [candidate["id"] for candidate, _ in rows]


def ranks(ordered: list[str], sites: list[str]) -> dict:
    """Per-site rank (1-based) and the first-hit rank; misses are len+1."""
    size = len(ordered)
    per_site = []
    for site in sites:
        if site in ordered:
            per_site.append(ordered.index(site) + 1)
        else:
            per_site.append(size + 1)
    first = min(per_site) if per_site else None
    return {"sites": per_site, "first": first}


# --- model ------------------------------------------------------------------

def cache_key(iid: str, config: str, k: int, tail: int, model: str, body_hash: str) -> str:
    return hashlib.sha256(f"{iid}|{config}|{k}|{tail}|{model}|{body_hash}".encode()).hexdigest()[:24]


def ask_server(base_url: str, body: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.rows: dict[str, dict] = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.rows[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, value: dict) -> None:
        self.rows[key] = {"key": key, **value}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(self.rows[key]) + "\n")


def run_config(
    item: dict,
    config: str,
    k: int,
    tail: int,
    base_url: str,
    cache: Cache,
    model_name: str,
) -> dict | None:
    failure = item["failure"] if config in ("L", "L1") else ""
    task = item["problem"] if config in ("L", "L0") else ""
    ranked = lc.rank_candidates(item["candidates"], failure, task, k, item["repo"])
    shortlist = ranked["ranked"]
    if not shortlist:
        return None
    # Frames and retrieval see the raw output; the model sees a focused window,
    # mirroring the Host (it receives raw pytest text and focuses it before asking).
    wire_failure = lc.focus_failure(failure, tail) if failure else ""
    body = lc.build_choice(shortlist, task, wire_failure, 0)
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    key = cache_key(item["instance_id"], config, k, tail, model_name, body_hash)
    cached = cache.get(key)
    started = time.time()
    if cached is not None:
        probabilities = cached["probabilities"]
        elapsed = cached.get("ms", 0.0)
    else:
        try:
            payload = ask_server(base_url, body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            print(f"    {config}: server error {error}", flush=True)
            return None
        probabilities = (payload.get("answers") or {}).get("loc", {}).get("probabilities") or {}
        if not probabilities:
            return None
        elapsed = round((time.time() - started) * 1000, 1)
        cache.put(key, {"probabilities": probabilities, "ms": elapsed})
    ordered = [cid for cid, _ in lc.rank_from_probabilities(shortlist, probabilities)]
    shortlist_ids = [candidate["id"] for candidate in shortlist]
    gold_sites = item["gold"]["sites"]
    return {
        "ranks": ranks(ordered, gold_sites),
        "shortlist": shortlist_ids,
        "stratum": ranked["stratum"],
        "frames": ranked["frames"],
        "ms": elapsed,
        "top": ordered[0] if ordered else None,
    }


# --- metrics ----------------------------------------------------------------

def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def summarize(rows: list[dict], key: str) -> dict:
    """Top-k, MRR and retrieval recall over site-level ranks."""
    site_ranks = [rank for row in rows for rank in row[key]["sites"]]
    firsts = [row[key]["first"] for row in rows if row[key]["first"] is not None]
    return {
        "instances": len(rows),
        "sites": len(site_ranks),
        "top1": mean([1.0 if rank <= 1 else 0.0 for rank in site_ranks]),
        "top3": mean([1.0 if rank <= 3 else 0.0 for rank in site_ranks]),
        "mrr": mean([1.0 / rank for rank in site_ranks]),
        "instance_top1": mean([1.0 if first <= 1 else 0.0 for first in firsts]),
        "instance_top3": mean([1.0 if first <= 3 else 0.0 for first in firsts]),
    }


def bootstrap_delta(rows: list[dict], left: str, right: str, resamples: int = 10000, seed: int = 23) -> dict:
    """Paired instance bootstrap over site-level MRR difference."""
    pairs = []
    for row in rows:
        if left not in row or right not in row:
            continue
        l = mean([1.0 / rank for rank in row[left]["sites"]])
        r = mean([1.0 / rank for rank in row[right]["sites"]])
        pairs.append(l - r)
    if not pairs:
        return {"delta": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    generator = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [generator.choice(pairs) for _ in pairs]
        deltas.append(sum(sample) / len(sample))
    deltas.sort()
    return {
        "delta": round(sum(pairs) / len(pairs), 3),
        "low": round(deltas[int(0.025 * len(deltas))], 3),
        "high": round(deltas[int(0.975 * len(deltas)) - 1], 3),
        "n": len(pairs),
    }


def shortlist_stats(instances: list[dict], k: int, mode: str = "both") -> dict:
    """Retrieval recall at k, over the same shortlist rule the Host will use."""
    recall_sites = []
    recall_instances = []
    strata = {"frame": 0, "assert-only": 0, "none": 0}
    for item in instances:
        failure = item["failure"] if mode in ("both", "failure") else ""
        task = item["problem"] if mode in ("both", "task") else ""
        ranked = lc.rank_candidates(item["candidates"], failure, task, k, item["repo"])
        shortlist_ids = {candidate["id"] for candidate in ranked["ranked"]}
        sites = item["gold"]["sites"]
        strata[ranked["stratum"]] = strata.get(ranked["stratum"], 0) + 1
        recall_sites.extend(1.0 if site in shortlist_ids else 0.0 for site in sites)
        recall_instances.append(1.0 if any(site in shortlist_ids for site in sites) else 0.0)
    return {
        "sites": mean(recall_sites),
        "instances": mean(recall_instances),
        "strata": strata,
        "n_sites": len(recall_sites),
    }


# --- main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/laya-ft/failure-loc")
    parser.add_argument("--configs", default="L,L0,L1")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--tail", type=int, default=2000)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    configs = [part.strip() for part in args.configs.split(",") if part.strip()]
    instances = discover(args.root)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        instances = [item for item in instances if item["instance_id"] in wanted]
    if args.limit:
        instances = instances[:args.limit]
    if not instances:
        print(f"no usable instances under {args.root}", flush=True)
        return 1

    health = None
    try:
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as response:
            health = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass
    model_name = json.dumps(health, sort_keys=True) if health else "unknown"
    print(f"instances={len(instances)} k={args.k} tail={args.tail} server={model_name}", flush=True)

    cache = Cache(os.path.join(args.root, "deepseek-cache.jsonl"))
    results = []
    for index, item in enumerate(instances, 1):
        row = {"instance_id": item["instance_id"], "sites": item["gold"]["sites"]}
        for config in configs:
            outcome = run_config(item, config, args.k, args.tail, args.base_url, cache, model_name)
            if outcome is None:
                continue
            row[config] = outcome
            ranked = lc.rank_candidates(
                item["candidates"],
                item["failure"] if config in ("L", "L1") else "",
                item["problem"] if config in ("L", "L0") else "",
                args.k,
                item["repo"],
            )
            shortlist = ranked["ranked"]
            row.setdefault("stratum", outcome["stratum"])
            row.setdefault("shortlist", outcome["shortlist"])
            if "H1" not in row:
                row["H1"] = {"ranks": ranks(order_h1(shortlist, ranked["frames"], item["failure"], item["problem"]), item["gold"]["sites"]), "shortlist": outcome["shortlist"], "ms": 0.0}
                row["H2"] = {"ranks": ranks(order_h2(shortlist, item["failure"], item["problem"]), item["gold"]["sites"]), "shortlist": outcome["shortlist"], "ms": 0.0}
                row["H3"] = {"ranks": ranks(outcome["shortlist"], item["gold"]["sites"]), "shortlist": outcome["shortlist"], "ms": 0.0}
        firsts = {
            key: row[key]["ranks"]["first"]
            for key in ("L", "L0", "L1", "H1", "H2", "H3")
            if key in row and isinstance(row[key], dict) and "ranks" in row[key]
        }
        results.append(row)
        print(f"[{index}/{len(instances)}] {item['instance_id'][:48]:48s} {row.get('stratum','?'):11s} " + " ".join(f"{key}={value}" for key, value in firsts.items()), flush=True)

    # Flatten ranks for the summarizer.
    flat = []
    for row in results:
        entry = {"instance_id": row["instance_id"], "sites": row["sites"], "stratum": row.get("stratum"), "shortlist": row.get("shortlist") or []}
        for key in ("L", "L0", "L1", "H1", "H2", "H3"):
            if key in row and isinstance(row[key], dict) and "ranks" in row[key]:
                entry[key] = row[key]["ranks"]
        flat.append(entry)

    summary = {}
    for key in ("L", "L0", "L1", "H1", "H2", "H3"):
        selected = [row for row in flat if key in row]
        if not selected:
            continue
        entry = summarize(selected, key)
        if key in ("L", "L0", "L1"):
            latencies = [
                result[key]["ms"] for result in results
                if key in result and isinstance(result[key], dict) and "ranks" in result[key]
            ]
            entry["latency_ms"] = mean(latencies)
        summary[key] = entry
        for stratum in ("frame", "assert-only"):
            subset = [row for row in selected if row.get("stratum") == stratum]
            if subset:
                summary[key][stratum] = summarize(subset, key)

    deltas = {}
    for left, right in (("L", "H3"), ("L", "H1"), ("L", "L0"), ("L", "L1")):
        deltas[f"{left}-{right}"] = bootstrap_delta(flat, left, right, args.bootstrap)

    retrieval = {
        "all": shortlist_stats(instances, args.k, "both"),
        "task_only": shortlist_stats(instances, args.k, "task"),
        "failure_only": shortlist_stats(instances, args.k, "failure"),
    }
    gates = evaluate_gates(summary, deltas, retrieval, args.k)

    payload = {
        "root": os.path.abspath(args.root),
        "server": health,
        "k": args.k,
        "tail": args.tail,
        "configs": configs,
        "instances": len(results),
        "summary": summary,
        "deltas": deltas,
        "retrieval": retrieval,
        "gates": gates,
        "rows": flat,
    }
    print(json.dumps({key: payload[key] for key in ("instances", "summary", "deltas", "retrieval", "gates")}, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"-> {args.out}")
    return 0


def evaluate_gates(summary: dict, deltas: dict, retrieval: dict, k: int) -> dict:
    """The four gates from docs/failure-localization-plan.md §5."""
    recall = retrieval["all"]["sites"]
    l_top1 = summary.get("L", {}).get("top1", 0.0)
    h3_top1 = summary.get("H3", {}).get("top1", 0.0)
    frame = summary.get("L", {}).get("frame", {})
    assert_only = summary.get("L", {}).get("assert-only", {})
    baseline_assert = summary.get("H3", {}).get("assert-only", {})
    wins = deltas.get("L-H3", {}).get("delta", 0.0)
    gate_retrieval = recall >= 0.85
    gate_ranking = (l_top1 - h3_top1) >= 0.10 and deltas.get("L-H3", {}).get("low", -1.0) > 0.0
    gate_signal = deltas.get("L-L0", {}).get("delta", 0.0) >= 0.10
    gate_niche = (
        bool(assert_only)
        and assert_only.get("top1", 0.0) >= 0.40
        and baseline_assert.get("top1", 1.0) <= 0.20
    )
    return {
        "recall_at_k": {"value": recall, "k": k, "pass": gate_retrieval},
        "ranking": {"delta_top1": round(l_top1 - h3_top1, 3), "delta_mrr": wins, "pass": gate_ranking},
        "failure_signal": {"delta_top1": deltas.get("L-L0", {}).get("delta", 0.0), "pass": gate_signal},
        "niche": {
            "assert_only_laya_top1": assert_only.get("top1"),
            "assert_only_h3_top1": baseline_assert.get("top1"),
            "pass": gate_niche,
        },
        "all_pass": bool(gate_retrieval and gate_ranking and gate_signal and gate_niche),
        "frame_instances": frame.get("instances", 0),
        "assert_only_instances": assert_only.get("instances", 0),
    }


if __name__ == "__main__":
    sys.exit(main())
