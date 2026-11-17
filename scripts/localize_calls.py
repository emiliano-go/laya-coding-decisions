"""Call-shape experiments for failure localization.

Same corpus, same shortlists, different ways of asking Laya:

    choice_long        the body used in the first probe (full ids as labels)
    choice_short       short labels h1..hN, description as the value
    choice_short_code  short labels, description replaced by a code snippet
    pointwise          one call per candidate (noul), body in the state
    pairwise           one call per candidate pair (2-way choice), Bradley-Terry

The point is to separate "the model cannot rank" from "the question shape hid
the evidence": a choice question shares `head_max_len` (192 tokens) across all
options, so a full path label can consume the whole per-option budget.

    python scripts/localize_calls.py --modes choice_long,choice_short --k 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import localize_candidates as lc  # noqa: E402
import localize_failure as lf  # noqa: E402

INSTRUCTIONS = "A test is failing. Which candidate most likely contains the bug?"
POINTWISE = "Does this code most likely contain the bug that makes the failing test fail?"
MAX_CODE = 700


def snippet(item: dict, candidate: dict, limit: int = MAX_CODE) -> str:
    """The candidate's own source, whitespace-collapsed, capped."""
    path = os.path.join(item["repo"], candidate["path"])
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return ""
    body = "\n".join(lines[candidate["line"] - 1:candidate["end"]])
    return body[:limit]


def call(base_url: str, body: dict, timeout: float = 180.0) -> dict:
    request = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask_choice(base_url: str, state: dict, criteria: dict, instructions: str = INSTRUCTIONS) -> dict:
    payload = call(base_url, {"state": state, "questions": {"loc": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}})
    return payload["answers"]["loc"].get("probabilities") or {}


def ask_noul(base_url: str, state: dict, instructions: str = POINTWISE) -> float:
    payload = call(base_url, {"state": state, "questions": {"loc": {
        "type": "noul", "instructions": instructions}}})
    return float(payload["answers"]["loc"].get("noul") or 0.0)


def base_state(item: dict, failure_chars: int) -> dict:
    return {
        "task": item["problem"][:400],
        "changes": [{"tool": "pytest", "output": lc.focus_failure(item["failure"], failure_chars)}],
    }


def ordered_ids(probabilities: dict, mapping: dict[str, str], shortlist: list[dict]) -> list[str]:
    rows = [(mapping.get(key, key), value) for key, value in probabilities.items()]
    rows.sort(key=lambda row: -row[1])
    seen = {ident for ident, _ in rows}
    # Anything the server dropped from the answer keeps its shortlist position.
    rows.extend((candidate["id"], 0.0) for candidate in shortlist if candidate["id"] not in seen)
    return [ident for ident, _ in rows]


def run_choice(item: dict, shortlist: list[dict], base_url: str, mode: str) -> list[str]:
    if mode == "choice_long":
        criteria = {candidate["id"]: candidate["text"] for candidate in shortlist}
        mapping = {candidate["id"]: candidate["id"] for candidate in shortlist}
        state = base_state(item, 2000)
    elif mode == "choice_short":
        criteria = {f"h{index + 1}": candidate["text"] for index, candidate in enumerate(shortlist)}
        mapping = {f"h{index + 1}": candidate["id"] for index, candidate in enumerate(shortlist)}
        state = base_state(item, 2000)
    elif mode == "choice_short_code":
        criteria = {
            f"h{index + 1}": (candidate["qualname"] + " :: " + " ".join(snippet(item, candidate, 150).split()))[:150]
            for index, candidate in enumerate(shortlist)
        }
        mapping = {f"h{index + 1}": candidate["id"] for index, candidate in enumerate(shortlist)}
        state = base_state(item, 2000)
    else:
        raise ValueError(mode)
    probabilities = ask_choice(base_url, state, criteria)
    return ordered_ids(probabilities, mapping, shortlist)


def run_pointwise(item: dict, shortlist: list[dict], base_url: str, workers: int) -> list[str]:
    failure = lc.focus_failure(item["failure"], 600)

    def score(candidate: dict) -> tuple[str, float]:
        state = {
            "task": item["problem"][:200],
            "changes": [
                {"tool": "read", "path": candidate["path"], "output": snippet(item, candidate, 700)},
                {"tool": "pytest", "output": failure},
            ],
        }
        return candidate["id"], ask_noul(base_url, state)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(score, shortlist))
    rows.sort(key=lambda row: -row[1])
    return [ident for ident, _ in rows]


def run_pairwise(item: dict, shortlist: list[dict], base_url: str, workers: int) -> list[str]:
    failure = lc.focus_failure(item["failure"], 500)
    pairs = [
        (left, right)
        for index, left in enumerate(shortlist)
        for right in shortlist[index + 1:]
    ]

    def compare(pair: tuple[dict, dict]) -> tuple[str, str, float]:
        left, right = pair
        state = {
            "task": item["problem"][:200],
            "changes": [
                {"tool": "read", "path": left["path"], "output": snippet(item, left, 550)},
                {"tool": "read", "path": right["path"], "output": snippet(item, right, 550)},
                {"tool": "pytest", "output": failure},
            ],
        }
        criteria = {"a": left["qualname"], "b": right["qualname"]}
        probabilities = ask_choice(base_url, state, criteria)
        return left["id"], right["id"], float(probabilities.get("a", 0.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compare, pairs))

    wins = {candidate["id"]: 0.0 for candidate in shortlist}
    games = {candidate["id"]: 0.0 for candidate in shortlist}
    for left_id, right_id, probability_left in results:
        games[left_id] += 1
        games[right_id] += 1
        if probability_left >= 0.5:
            wins[left_id] += 1
        else:
            wins[right_id] += 1
    # Bradley-Terry by minorisation-maximisation; wins-only is unstable at k=8.
    strength = {ident: 0.5 for ident in wins}
    for _ in range(30):
        updated = {}
        for ident in wins:
            denominator = 0.0
            for left_id, right_id, probability_left in results:
                if ident not in (left_id, right_id):
                    continue
                other = right_id if ident == left_id else left_id
                denominator += 1.0 / (strength[ident] + strength[other])
            updated[ident] = (wins[ident] + 0.5) / denominator if denominator else 0.0
        total = sum(updated.values()) or 1.0
        strength = {ident: value / total for ident, value in updated.items()}
    ordered = sorted(strength.items(), key=lambda row: -row[1])
    return [ident for ident, _ in ordered]


MODES = ("choice_long", "choice_short", "choice_short_code", "pointwise", "pairwise")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/laya-ft/failure-loc")
    parser.add_argument("--modes", default="choice_long,choice_short,choice_short_code")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        parser.error(f"unknown modes: {unknown}")

    instances = lf.discover(args.root)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        instances = [item for item in instances if item["instance_id"] in wanted]
    if args.limit:
        instances = instances[:args.limit]
    if not instances:
        print(f"no usable instances under {args.root}", flush=True)
        return 1
    print(f"instances={len(instances)} k={args.k} modes={modes} workers={args.workers}", flush=True)

    rows = []
    started = time.time()
    for index, item in enumerate(instances, 1):
        ranked = lc.rank_candidates(item["candidates"], item["failure"], item["problem"], args.k, item["repo"])
        shortlist = ranked["ranked"]
        shortlist_ids = [candidate["id"] for candidate in shortlist]
        row = {
            "instance_id": item["instance_id"],
            "stratum": ranked["stratum"],
            "sites": item["gold"]["sites"],
            "shortlist": shortlist_ids,
        }
        for mode in modes:
            began = time.time()
            try:
                if mode == "pointwise":
                    ordered = run_pointwise(item, shortlist, args.base_url, args.workers)
                elif mode == "pairwise":
                    ordered = run_pairwise(item, shortlist, args.base_url, args.workers)
                else:
                    ordered = run_choice(item, shortlist, args.base_url, mode)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
                print(f"  {mode}: {error}", flush=True)
                continue
            row[mode] = {
                "ordered": ordered,
                "ranks": lf.ranks(ordered, item["gold"]["sites"]),
                "ms": round((time.time() - began) * 1000, 1),
            }
        # Deterministic control on the same shortlist.
        row["H3"] = {"ranks": lf.ranks(shortlist_ids, item["gold"]["sites"]), "ms": 0.0}
        rows.append(row)
        firsts = {mode: row[mode]["ranks"]["first"] for mode in modes if mode in row}
        print(f"[{index}/{len(instances)}] {item['instance_id'][:44]:44s} "
              + " ".join(f"{mode}={value}" for mode, value in firsts.items()), flush=True)

    flat = []
    for row in rows:
        entry = {key: row[key] for key in ("instance_id", "stratum", "sites", "shortlist")}
        for mode in modes + ["H3"]:
            if mode in row:
                entry[mode] = row[mode]["ranks"]
        flat.append(entry)

    summary = {}
    for mode in modes + ["H3"]:
        selected = [row for row in flat if mode in row]
        if not selected:
            continue
        entry = lf.summarize(selected, mode)
        entry["latency_ms"] = lf.mean([row[mode]["ms"] for row in rows if mode in row])
        summary[mode] = entry
        conditional = [
            row for row in selected
            if any(site in row["shortlist"] for site in row["sites"])
        ]
        if conditional:
            summary[mode]["conditional"] = lf.summarize(conditional, mode)

    payload = {
        "root": os.path.abspath(args.root),
        "k": args.k,
        "modes": modes,
        "instances": len(rows),
        "server_seconds": round(time.time() - started, 1),
        "summary": summary,
        "rows": flat,
    }
    print(json.dumps({mode: summary.get(mode) for mode in modes + ["H3"]}, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
