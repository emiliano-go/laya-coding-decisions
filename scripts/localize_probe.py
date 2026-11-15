"""Zero-shot function-localization probe for Laya.

For each SWE-smith instance, parse the repo, derive the buggy function from the
breaking patch, build a shortlist of candidate functions from the issue text
(issue-only; the patch is never sent), and ask Laya to rank them.

  python scripts/localize_probe.py --limit 20 --topk 10
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import urllib.request
from collections import Counter


ADDED = re.compile(r"^\+(?!\+\+)")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def patch_added_lines(patch: str) -> dict[str, set[int]]:
    files: dict[str, set[int]] = {}
    current = ""
    line = 0
    for raw in (patch or "").splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].strip()
            current = name[2:] if name.startswith("b/") else name
            files.setdefault(current, set())
            continue
        match = HUNK.match(raw)
        if match:
            line = int(match.group(1))
            continue
        if current and ADDED.match(raw):
            files[current].add(line)
            line += 1
        elif current and raw.startswith(" "):
            line += 1
    return {path: lines for path, lines in files.items() if lines}


def functions(path: str, rel: str) -> list[dict]:
    try:
        tree = ast.parse(open(path, errors="replace").read())
    except (SyntaxError, OSError):
        return []
    out = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                doc = ast.get_docstring(child) or ""
                first = doc.strip().splitlines()[0][:60] if doc else ""
                args = ", ".join(a.arg for a in child.args.args if a.arg not in ("self", "cls"))[:40]
                body = ""
                if child.body and not (isinstance(child.body[0], ast.Expr) and isinstance(getattr(child.body[0], "value", None), ast.Constant) and isinstance(child.body[0].value.value, str)):
                    try:
                        body = ast.unparse(child.body[0]).replace("\n", " ")[:50]
                    except Exception:
                        body = ""
                hint = first or body
                out.append({
                    "id": f"{rel}:{name}",
                    "text": f"{rel}:{name}({args})" + (f" — {hint}" if hint else ""),
                    "line": child.lineno,
                    "end": getattr(child, "end_lineno", child.lineno),
                })
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower())


def shortlist(problem: str, candidates: list[dict], topk: int) -> list[dict]:
    query = Counter(tokens(problem))
    if not query:
        return candidates[:topk]
    scored = []
    for candidate in candidates:
        body = Counter(tokens(candidate["text"]))
        score = sum(min(count, body[token]) * math.log(1 + len(body) / (1 + body[token])) for token, count in query.items())
        scored.append((score, candidate))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [candidate for _, candidate in scored[:topk]]


def ask(base_url: str, problem: str, candidates: list[dict]) -> dict[str, float]:
    criteria = {candidate["id"]: candidate["text"] for candidate in candidates}
    body = json.dumps({
        "state": {"task": problem},
        "questions": {
            "loc": {
                "type": "choice",
                "instructions": "Which function most likely contains the bug described by the task?",
                "criteria": criteria,
            },
        },
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/systemone", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read())
    return payload["answers"]["loc"]["probabilities"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="/data/laya-ft/swe-smith/tasks.json")
    parser.add_argument("--root", default="/data/laya-ft/swe-smith")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    tasks = json.load(open(args.tasks))[: args.limit]
    ranks = []
    recall = 0
    rows = []
    for index, summary in enumerate(tasks, 1):
        iid = summary["instance_id"]
        task_path = os.path.join(args.root, iid, "task.json")
        task = json.load(open(task_path)) if os.path.exists(task_path) else summary
        patch = task.get("patch") or ""
        problem = task.get("problem_statement") or summary.get("problem_statement") or ""
        repo = os.path.join(args.root, iid, "repo")
        if not patch or not problem or not os.path.isdir(repo):
            continue
        added = patch_added_lines(patch)
        candidates = []
        for base, _dirs, files in os.walk(repo):
            if "/.git" in base or "/node_modules" in base:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(base, name)
                rel = os.path.relpath(path, repo)
                candidates.extend(functions(path, rel))
        if len(candidates) < 2:
            continue
        gold = None
        gold_count = 0
        for path, lines in added.items():
            rel = path
            for candidate in candidates:
                if candidate["id"].split(":")[0] != rel:
                    continue
                hits = sum(1 for line in lines if candidate["line"] <= line <= candidate["end"])
                if hits > gold_count:
                    gold, gold_count = candidate["id"], hits
        if gold is None:
            continue
        short = shortlist(problem, candidates, args.topk)
        in_short = any(candidate["id"] == gold for candidate in short)
        recall += int(in_short)
        if not in_short:
            continue
        probabilities = ask(args.base_url, problem, short)
        ordered = sorted(probabilities.items(), key=lambda row: row[1], reverse=True)
        rank = next((i for i, (key, _) in enumerate(ordered, 1) if key == gold), None)
        if rank is None:
            continue
        ranks.append(rank)
        rows.append({"instance_id": iid, "gold": gold, "rank": rank, "shortlist": len(short), "top": ordered[0][0], "top_prob": ordered[0][1]})
        print(f"[{index}/{len(tasks)}] rank {rank}/{len(short)} gold={gold}", flush=True)
    n = len(ranks)
    print(json.dumps({
        "tasks": len(tasks),
        "evaluated": n,
        "retrieval_recall": round(recall / max(1, len(tasks)), 3),
        "top1": round(sum(1 for r in ranks if r == 1) / max(1, n), 3),
        "top5": round(sum(1 for r in ranks if r <= 5) / max(1, n), 3),
        "mrr": round(sum(1 / r for r in ranks) / max(1, n), 3),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
