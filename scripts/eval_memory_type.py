"""Memory type classification: Laya vs cheap baselines.

Reads every project's Totem store, scrubs type leakage, and compares three
predictors on the same 490 memories:

  B0  majority class
  B1  TF-IDF nearest neighbours (stdlib only)
  B2  potion embedding nearest neighbours (optional, see --embedding)
  L    Laya, one 8-way choice per memory

Two settings: in-project leave-one-out (headline) and cross-project.

    python scripts/eval_memory_type.py --root /home/eclipse/Documents/GitHub
    python scripts/eval_memory_type.py --skip-laya --report-only

Artifacts go to `.work/` (gitignored by the `*.jsonl` rule).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TYPES = {
    "implementation": "what a piece of code does: an API, function, schema, or file structure",
    "gotcha": "non-obvious behaviour, or an error that bites later",
    "decision": "a choice that was made, and why",
    "architecture": "how a component or module is structured",
    "observation": "a measured fact about the environment or runtime",
    "bug": "a defect that was found",
    "invariant": "a rule that must always hold",
    "contract": "an interface guarantee between components",
}
TYPE_NAMES = tuple(TYPES)
INSTRUCTIONS = "Which type best describes this engineering memory?"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


# --- collection -------------------------------------------------------------

def discover(root: str) -> list[tuple[str, str]]:
    """(project_dir, totem_db) for every project under `root`."""
    found = []
    for db in glob.glob(os.path.join(root, "*", ".totem", "totem.db")):
        project = os.path.dirname(os.path.dirname(db))
        found.append((project, db))
    return sorted(found)


def collect(root: str, totem: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for project, _db in discover(root):
        try:
            proc = subprocess.run(
                [totem, "--project", project, "export"],
                capture_output=True, text=True, timeout=180,
            )
            payload = json.loads(proc.stdout or "{}")
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as error:
            print(f"  skip {os.path.basename(project)}: {error}", flush=True)
            continue
        for item in payload.get("items", []):
            if item.get("status") != "active":
                continue
            ident = str(item.get("id") or "")
            if not ident or ident in seen:
                continue
            kind = str(item.get("type") or "")
            if kind not in TYPES:
                continue
            seen.add(ident)
            raw_tags = [str(tag) for tag in (item.get("tags") or [])]
            leak_tags = any(tag.lower() in TYPES for tag in raw_tags)
            tags = [tag for tag in raw_tags if tag.lower() not in TYPES]
            title = str(item.get("title") or "")[:200]
            rows.append({
                "id": ident,
                "project": os.path.basename(project),
                "type": kind,
                "title": title,
                "statement": str(item.get("statement") or "")[:1500],
                "tags": tags,
                "leak_tags": leak_tags,
                "leak_title": kind in title.lower().split(),
            })
    return rows


# --- baselines --------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def document(row: dict) -> str:
    return f"{row['title']}\n{row['statement']}"


class Tfidf:
    def __init__(self, docs: list[str]):
        self.df: collections.Counter = collections.Counter()
        self.vectors: list[dict[str, float]] = []
        total = len(docs)
        for doc in docs:
            for term in set(tokenise(doc)):
                self.df[term] += 1
        for doc in docs:
            counts = collections.Counter(tokenise(doc))
            if not counts:
                self.vectors.append({})
                continue
            weights = {
                term: (1 + math.log(count)) * (math.log((1 + total) / (1 + self.df[term])) + 1)
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            self.vectors.append({term: value / norm for term, value in weights.items()})

    def cosine(self, left: int, right: int, other: dict[str, float] | None = None) -> float:
        first = self.vectors[left]
        second = self.vectors[right] if other is None else other
        if len(first) > len(second):
            first, second = second, first
        return sum(value * second.get(term, 0.0) for term, value in first.items())


class Dense:
    """Normalised dense vectors with the same `.cosine` surface as Tfidf."""

    def __init__(self, matrix):
        self.matrix = matrix

    def cosine(self, left: int, right: int) -> float:
        return float(self.matrix[left] @ self.matrix[right])


def dense_vectors(documents: list[str], model_name: str):
    """potion embeddings; raises ImportError when model2vec is unavailable."""
    import numpy as np
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(model_name)
    matrix = np.asarray(model.encode(documents), dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Dense(matrix / norms)


def predict_knn(rows: list[dict], vectors, setting: str, k: int = 5) -> list[str]:
    """Nearest-neighbour vote; `setting` is in-project or cross."""
    by_project: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_project[row["project"]].append(index)
    global_counts = collections.Counter(row["type"] for row in rows)
    fallback = global_counts.most_common(1)[0][0]
    predictions = []
    for index, row in enumerate(rows):
        same = [i for i in by_project[row["project"]] if i != index]
        pool = same if setting == "in-project" else [i for i in range(len(rows)) if i != index]
        if not pool:
            predictions.append(fallback)
            continue
        scored = sorted(pool, key=lambda other: -vectors.cosine(index, other))
        neighbours = scored[: min(k, len(scored))]
        votes = collections.Counter(rows[other]["type"] for other in neighbours)
        best = max(votes.values())
        winners = [label for label, count in votes.items() if count == best]
        predictions.append(winners[0] if len(winners) == 1 else rows[neighbours[0]]["type"])
    return predictions


def predict_majority(rows: list[dict], setting: str) -> list[str]:
    by_project: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        by_project[row["project"]][row["type"]] += 1
    global_counts = collections.Counter(row["type"] for row in rows)
    predictions = []
    for row in rows:
        if setting == "in-project":
            counts = by_project[row["project"]].copy()
            counts[row["type"]] -= 1
            counts = +counts
            if not counts:
                counts = global_counts
        else:
            counts = global_counts
        predictions.append(counts.most_common(1)[0][0])
    return predictions


# --- Laya -------------------------------------------------------------------

def ask(base_url: str, state: dict, criteria: dict) -> dict[str, float]:
    body = {"state": state, "questions": {"type": {
        "type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}}
    request = urllib.request.Request(
        f"{base_url}/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["answers"]["type"].get("probabilities") or {}


def run_laya(rows: list[dict], base_url: str, cache_path: str, workers: int = 4) -> dict[str, str]:
    cached: dict[str, str] = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cached[row["id"]] = row["predicted"]
    criteria = dict(TYPES)

    def one(row: dict) -> tuple[str, str]:
        if row["id"] in cached:
            return row["id"], cached[row["id"]]
        state = {"task": row["title"], "answer": row["statement"]}
        probabilities = ask(base_url, state, criteria)
        predicted = max(probabilities, key=probabilities.get) if probabilities else ""
        return row["id"], predicted

    pending = [row for row in rows if row["id"] not in cached]
    print(f"  laya: {len(cached)} cached, {len(pending)} to run", flush=True)
    with open(cache_path, "a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (ident, predicted) in enumerate(pool.map(one, pending), 1):
                cached[ident] = predicted
                out.write(json.dumps({"id": ident, "predicted": predicted}) + "\n")
                out.flush()
                if index % 50 == 0:
                    print(f"    {index}/{len(pending)}", flush=True)
    return cached


# --- metrics ----------------------------------------------------------------

def scores(rows: list[dict], predictions: list[str]) -> dict:
    pairs = [(row["type"], predicted) for row, predicted in zip(rows, predictions)]
    per_class = {}
    for label in TYPE_NAMES:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": tp + fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3),
        }
    present = [label for label in TYPE_NAMES if per_class[label]["support"] > 0]
    macro = sum(per_class[label]["f1"] for label in present) / len(present) if present else 0.0
    total = len(pairs) or 1
    accuracy = sum(1 for gold, pred in pairs if gold == pred) / total
    weighted = sum(
        per_class[label]["f1"] * per_class[label]["support"] for label in present
    ) / total
    confusion = collections.Counter((gold, pred) for gold, pred in pairs if gold != pred)
    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro, 3),
        "weighted_f1": round(weighted, 3),
        "classes": per_class,
        "confusions": [
            {"gold": gold, "pred": pred, "n": count}
            for (gold, pred), count in confusion.most_common(8)
        ],
    }


def bootstrap(rows: list[dict], predictions: dict[str, list[str]], resamples: int = 10000, seed: int = 7) -> dict:
    projects = sorted({row["project"] for row in rows})
    by_project: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_project[row["project"]].append(index)
    generator = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in predictions}
    deltas: list[float] = []
    for _ in range(resamples):
        chosen = [generator.choice(projects) for _ in projects]
        indices = [index for project in chosen for index in by_project[project]]
        subset = [rows[index] for index in indices]
        values = {}
        for name, predicted in predictions.items():
            values[name] = scores(subset, [predicted[index] for index in indices])["macro_f1"]
            samples[name].append(values[name])
        if "laya" in values and "tfidf" in values:
            deltas.append(round(values["laya"] - values["tfidf"], 4))
    summary = {}
    for name, values in samples.items():
        values.sort()
        summary[name] = {
            "low": round(values[int(0.025 * len(values))], 3),
            "high": round(values[int(0.975 * len(values)) - 1], 3),
        }
    deltas.sort()
    summary["laya_minus_tfidf"] = {
        "mean": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "low": round(deltas[int(0.025 * len(deltas))], 3) if deltas else None,
        "high": round(deltas[int(0.975 * len(deltas)) - 1], 3) if deltas else None,
    }
    return summary


# --- report -----------------------------------------------------------------

def markdown(per_class: dict, delta: float | None, verdict: str, best: str, rows: list[dict], summaries_by_setting: dict, leakage: dict) -> str:
    lines = ["# Memory type classification — results", ""]
    lines.append(f"Corpus: {len(rows)} active memories over "
                 f"{len({row['project'] for row in rows})} projects. "
                 f"Types: {dict(collections.Counter(row['type'] for row in rows))}.")
    lines.append("")
    lines.append(f"Leakage scrub: {leakage['tags']} rows carried a type name in their tags "
                 f"(stripped); {leakage['titles']} titles contain the type word (flagged).")
    lines.append("")
    for setting, table in summaries_by_setting.items():
        lines.append(f"## {setting}")
        lines.append("")
        lines.append("| predictor | accuracy | macro-F1 | weighted F1 |")
        lines.append("|---|---:|---:|---:|")
        for name, values in table.items():
            lines.append(f"| {name} | {values['accuracy']} | {values['macro_f1']} | {values['weighted_f1']} |")
        lines.append("")
    if delta is not None:
        lines.append(f"**Laya − {best} macro-F1 (in-project): {delta:+.3f} → {verdict}** "
                     f"(bar: +0.10 and ≥ 0.55)")
        lines.append("")
    lines.append("## Per class (in-project)")
    lines.append("")
    lines.append("| type | support | precision | recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, values in per_class.items():
        lines.append(f"| {label} | {values['support']} | {values['precision']} | "
                     f"{values['recall']} | {values['f1']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/eclipse/Documents/GitHub")
    parser.add_argument("--work", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".work"))
    parser.add_argument("--totem", default=os.path.expanduser("~/.local/bin/totem"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--skip-laya", action="store_true")
    parser.add_argument("--embedding", action="store_true", help="also score potion-retrieval-32M kNN")
    parser.add_argument("--embedding-model", default="minishlab/potion-retrieval-32M")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.work, exist_ok=True)
    rows_path = os.path.join(args.work, "memory-types.jsonl")
    laya_path = os.path.join(args.work, "memory-type-laya.jsonl")

    if not args.report_only:
        rows = collect(args.root, args.totem)
        with open(rows_path, "w") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        print(f"collected {len(rows)} rows -> {rows_path}", flush=True)
    else:
        rows = [json.loads(line) for line in open(rows_path)]

    leakage = {
        "tags": sum(1 for row in rows if row.get("leak_tags")),
        "titles": sum(1 for row in rows if row.get("leak_title")),
    }
    print(f"leakage: tags={leakage['tags']} titles={leakage['titles']}", flush=True)

    documents = [document(row) for row in rows]
    vectors = Tfidf(documents)
    predictions: dict[str, list[str]] = {}
    by_setting: dict[str, dict] = {}

    dense = None
    if args.embedding:
        try:
            dense = dense_vectors(documents, args.embedding_model)
            print("embedding: potion-retrieval-32M", flush=True)
        except ImportError as error:
            print(f"embedding unavailable ({error}); falling back to TF-IDF only", flush=True)

    for setting in ("in-project", "cross-project"):
        predictions[f"majority_{setting}"] = predict_majority(rows, setting)
        predictions[f"tfidf_{setting}"] = predict_knn(rows, vectors, setting)
        if dense is not None:
            predictions[f"embedding_{setting}"] = predict_knn(rows, dense, setting)

    if not args.skip_laya:
        laya = run_laya(rows, args.base_url, laya_path, args.workers)
        predictions["laya"] = [laya.get(row["id"], "") for row in rows]

    for setting in ("in-project", "cross-project"):
        table = {
            "majority": scores(rows, predictions[f"majority_{setting}"]),
            "tfidf": scores(rows, predictions[f"tfidf_{setting}"]),
        }
        if f"embedding_{setting}" in predictions:
            table["embedding"] = scores(rows, predictions[f"embedding_{setting}"])
        if "laya" in predictions:
            table["laya"] = scores(rows, predictions["laya"])
        by_setting[setting] = table

    headline = by_setting["in-project"]
    headline_names = {name: predictions[f"{name}_in-project" if name != "laya" else "laya"] for name in headline}
    interval = bootstrap(rows, headline_names, args.bootstrap) if args.bootstrap else {}

    print(json.dumps({"in_project": headline, "cross_project": by_setting["cross-project"],
                      "bootstrap": interval}, indent=1))

    detail = headline.get("laya") or headline["tfidf"]
    best = max((name for name in ("tfidf", "embedding") if name in headline),
               key=lambda name: headline[name]["macro_f1"])
    delta = None
    verdict = "n/a"
    if "laya" in headline:
        delta = round(headline["laya"]["macro_f1"] - headline[best]["macro_f1"], 3)
        verdict = "PASS" if delta >= 0.10 and headline["laya"]["macro_f1"] >= 0.55 else "FAIL"
    text = markdown(detail["classes"], delta, verdict, best, rows, by_setting, leakage)
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "memory-type-results.md")
    with open(doc, "w") as out:
        out.write(text + "\n")
    print(f"-> {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
