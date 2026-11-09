"""Teacher-generated decision training data (DeepSeek v4.1 flash).

One API call per task returns:
  - 1-3 invariants/gotchas for the task (-> state.memory)
  - 2-3 independent semantic bug mutations  (-> bug_risk 2/3)
  - 1-2 scope-creep mutations               (-> no_unrelated_changes false)
plus the untouched correct patch as a positive control row. Every row for a task
carries the *same* invariant memory, so memory presence never predicts a label
(the M5 shortcut guard). Preference pairs (chosen=correct, rejected=mutated) are
emitted for the E3/GAGAR ranking objective at no extra API cost.

Labels are TEACHER labels (LLM-written), validated structurally only. Rows carry
"validated": "structural" and "teacher": "<model>"; execution verification is a
separate lane.

Usage:
  python3 gen_deepseek.py --n 20 --workers 8 --out /data/laya-ft/gen_deepseek
  python3 gen_deepseek.py --n 3000 --workers 32      # resumable
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_dataset_v2 import MAX_CHARS, QUESTIONS, noul, score, split_of  # noqa: E402

CACHE = os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "cache3")
SOURCES = ["gold_fixed", "swe_smith_traj", "swe_rebench_traj", "bigcodebench"]


def load_env():
    return (
        os.environ["DEEPSEEK_API_KEY"],
        os.environ.get("DEEPSEEK_BASE_URL", "https://model.inferx.net/endpoints/v1/chat/completions"),
        os.environ.get("DEEPSEEK_MODEL", "deepseek-v4.1-flash-1017-basic"),
    )


KEY, URL, MODEL = load_env()


SUSPECT = re.compile(r"#.*(incorrect|intentionally|injected|deliberate|sabotag|do not fix|leave.{0,10}broken)", re.I)


def suspect(diff):
    """Reject source patches that look like deliberate bug-injection patches."""
    added = "\n".join(l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    return bool(SUSPECT.search(added))


def clean_task(task):
    """swe_smith_traj stores a stringified list of content parts; unwrap it."""
    task = (task or "").strip()
    if task.startswith("[{") and "'type'" in task:
        try:
            parts = ast.literal_eval(task)
            task = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (ValueError, SyntaxError):
            pass
    return task.strip()[:600]


def task_key(task, diff):
    return hashlib.sha1((task + "\x00" + diff).encode("utf-8", "ignore")).hexdigest()[:16]


def extract_json(text):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no json object")
    return json.loads(text[start : end + 1])


def chat_json(system, user, temperature=0.8, max_tokens=6000, attempts=15):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    last = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            URL,
            data=data,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read())
            choice = payload["choices"][0]
            text = choice["message"].get("content")
            if not text:
                raise ValueError(f"empty content (finish_reason={choice.get('finish_reason')})")
            return extract_json(text)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError, TimeoutError, socket.timeout) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403):
                raise
            time.sleep(min(2 ** attempt, 60) + random.uniform(0, 5))
    raise RuntimeError(f"chat_json failed after {attempts}: {last}")


def valid_diff(diff, original):
    if not isinstance(diff, str):
        return False
    if len(diff.strip()) < 20 or diff.strip() == original.strip():
        return False
    return "@@" in diff and "\n+" in diff


SYSTEM = "You generate training data for a code-review model. Return only a JSON object."

# Set from CLI before the worker pool starts (read-only afterwards).
NBUGS = 2
NSCOPES = 1
FOCUS = ""

# Optional differentiated prompt families. Each shard can lean into a bug class
# so the pool is not dominated by the same shallow mutations.
FOCUS_INSTRUCTION = {
    "": "",
    "security": "Prefer security-relevant bugs: authz bypass, injection, path traversal, secret leakage, unsafe deserialization.",
    "concurrency": "Prefer concurrency bugs: races, missing locks, atomicity, TOCTOU, wrong synchronization scope.",
    "edge": "Prefer edge-case bugs: empty/null inputs, boundaries, unicode, overflow, empty collections, partial failure.",
    "resource": "Prefer resource-lifecycle bugs: leaks, missing cleanup, double-free, unclosed handles, wrong exception scope.",
}

PROMPT = """Task:
{task}

Correct patch (unified diff):
{diff}

Produce training variants of this patch.

1) invariants: 1-3 invariants, constraints, or gotchas a reviewer must check for this
   change (must not break, edge cases, prior regressions). Describe the TASK, never
   whether this particular patch is good or bad.
2) bugs: exactly {nbugs} INDEPENDENT bugs (return [] if 0), each in its own mutated copy of the patch. Each
   mutation injects exactly ONE realistic bug into added/changed lines. Use distinct
   classes from: off-by-one, wrong-operator, null/empty-handling, early-return,
   wrong-comparison, resource-leak, wrong-exception-scope, broken-invariant,
   type-coercion. Keep it syntactically valid and plausible; never annotate the bug
   in comments; never touch unrelated lines.
3) scopes: exactly {nscopes} scope-creep mutation(s) of the patch (return [] if 0; unrelated renames,
   reorder/reformat untouched code, speculative compatibility branch, swallowed
   exception, unrequested feature, broadened export). Requested behavior must still
   work.
{focus}
Return ONLY this JSON:
{{"invariants": [{{"type": "invariant|constraint|gotcha", "title": "<short>", "statement": "<max 160 chars>"}}],
 "bugs": [{{"mutated_diff": "<full unified diff>", "bug_class": "<class>", "severity": <1|2|3>, "requirement_impact": <true|false>, "explanation": "<max 200 chars>"}}],
 "scopes": [{{"mutated_diff": "<full unified diff>", "unrelated": ["<change>", ...], "risk": <0|1>, "explanation": "<max 200 chars>"}}]}}
severity: 1 subtle, 2 likely bug, 3 clearly broken. requirement_impact=true means the
task requirement is no longer met. Each mutated_diff must be a complete unified diff
that differs from the correct patch."""


ANSWER_QUALITY_Q = {
    "type": "score",
    "instructions": "Rate how completely and correctly the assistant's answer addresses the user's request.",
    "criteria": [
        "Wrong or missing: the answer does not address the request or is incorrect",
        "Partial: the answer addresses the request but omits important parts",
        "Mostly complete: addresses the request with only minor omissions",
        "Complete: fully and correctly addresses the user's request",
    ],
}

ANSWER_PROMPT = """Task:
{task}

Patch under discussion:
{diff}

Write a developer question about this change, then four candidate assistant answers of
decreasing quality. The question should ask what the change does or whether it is correct.

Return ONLY this JSON:
{{"question": "<one-line developer question>",
 "complete": "<answer that fully and correctly addresses it>",
 "mostly": "<correct but omits a minor detail>",
 "partial": "<addresses only part of it>",
 "wrong": "<plausible but incorrect>"}}
Each answer is 1-4 sentences, no code blocks."""


def gen_answer_task(task, diff):
    data = chat_json(SYSTEM, ANSWER_PROMPT.format(task=task, diff=diff), temperature=0.7, max_tokens=1200)
    question = str(data.get("question", "")).strip()
    answers = [
        (data.get("complete"), 3),
        (data.get("mostly"), 2),
        (data.get("partial"), 1),
        (data.get("wrong"), 0),
    ]
    return {"mode": "answer", "task": task, "diff": diff, "question": question, "answers": answers}


MEMORY_PROMPT = """Task:
{task}

Correct patch:
{diff}

1) invariants: 2-3 invariants, constraints, or gotchas for this change.
2) respected: a patch that satisfies every invariant (the correct patch is fine).
3) violations: 1-2 patches that each violate exactly ONE named invariant in a plausible
   way: the change must look locally reasonable, but break the stated contract. Say which
   invariant index each one violates.

Return ONLY this JSON:
{{"invariants": [{{"type": "invariant|constraint|gotcha", "title": "<short>", "statement": "<max 160 chars>"}}],
 "respected_diff": "<full unified diff>",
 "violations": [{{"mutated_diff": "<full unified diff>", "invariant_index": <int>, "severity": <2|3>, "explanation": "<max 200 chars>"}}]}}
Each mutated_diff must be a complete unified diff that differs from the correct patch."""


def gen_memory_task(task, diff):
    data = chat_json(SYSTEM, MEMORY_PROMPT.format(task=task, diff=diff), temperature=0.75, max_tokens=6000)
    tid = hashlib.sha1(task.encode("utf-8", "ignore")).hexdigest()[:8]
    memories = [
        {"id": f"{tid}{i}", "type": str(m.get("type", "invariant")), "title": str(m.get("title", ""))[:60], "statement": str(m.get("statement", ""))[:160]}
        for i, m in enumerate((data.get("invariants") or [])[:3])
        if isinstance(m, dict) and m.get("statement")
    ]
    respected = data.get("respected_diff")
    if not valid_diff(respected, diff):
        respected = diff
    seen = {diff.strip()}
    violations = []
    for v in (data.get("violations") or [])[:2]:
        if not isinstance(v, dict):
            continue
        d = v.get("mutated_diff", "")
        if valid_diff(d, diff) and d.strip() not in seen:
            seen.add(d.strip())
            violations.append(v)
    return {"mode": "memory", "task": task, "diff": respected, "memories": memories, "violations": violations}


BENIGN_PROMPT = """Task:
{task}

Correct patch:
{diff}

Produce 4 SMALL, correct, behavior-preserving changes that a reviewer should accept:
rename a local variable, add a clarifying comment, extract a helper with no behavior
change, add a guard/validation, or fix a typo. Each change must be a complete unified
diff against the same area and must not introduce a bug or unrelated work.

Return ONLY this JSON:
{{"changes": [{{"mutated_diff": "<full unified diff>", "kind": "<rename|comment|helper|guard|typo>"}}]}}"""

BENIGN_MEMORIES = [
    {"type": "gotcha", "title": "public contract stable", "statement": "Callers depend on the existing return shape and error behavior."},
    {"type": "constraint", "title": "stay in scope", "statement": "Keep changes limited to the requested behavior; no unrelated edits."},
    {"type": "invariant", "title": "tests stay green", "statement": "Existing tests must keep passing after the change."},
]


def gen_benign_task(task, diff):
    data = chat_json(SYSTEM, BENIGN_PROMPT.format(task=task, diff=diff), temperature=0.8, max_tokens=6000)
    changes = []
    for change in (data.get("changes") or [])[:4]:
        if isinstance(change, dict) and valid_diff(change.get("mutated_diff", ""), diff):
            changes.append(change)
    return {"mode": "benign", "task": task, "diff": diff, "changes": changes}


def gen_task(task, diff):
    data = chat_json(
        SYSTEM,
        PROMPT.format(
            task=task,
            diff=diff,
            nbugs=NBUGS,
            nscopes=NSCOPES,
            focus=("4) " + FOCUS_INSTRUCTION[FOCUS] + "\n") if FOCUS_INSTRUCTION.get(FOCUS) else "",
        ),
    )
    tid = hashlib.sha1(task.encode("utf-8", "ignore")).hexdigest()[:8]
    memories = [
        {"id": f"{tid}{i}", "type": str(m.get("type", "invariant")), "title": str(m.get("title", ""))[:60], "statement": str(m.get("statement", ""))[:160]}
        for i, m in enumerate((data.get("invariants") or [])[:3])
        if isinstance(m, dict) and m.get("statement")
    ]
    seen = {diff.strip()}
    bugs, scopes = [], []
    for b in data.get("bugs") or []:
        if not isinstance(b, dict):
            continue
        d = b.get("mutated_diff", "")
        if valid_diff(d, diff) and d.strip() not in seen:
            seen.add(d.strip())
            bugs.append(b)
    for s in data.get("scopes") or []:
        if not isinstance(s, dict):
            continue
        d = s.get("mutated_diff", "")
        if valid_diff(d, diff) and d.strip() not in seen:
            seen.add(d.strip())
            scopes.append(s)
    return {"task": task, "diff": diff, "memories": memories, "bugs": bugs[:3], "scopes": scopes[:2]}


def make_state(task, memories, diff):
    return {
        "task": task,
        "memory": memories,
        "changes": [{"tool": "apply_patch", "input": diff, "output": "applied"}],
        "answer": "applied the change",
        "toolCalls": [{"tool": "apply_patch", "title": "patch", "status": "completed"}],
    }


def emit(task, memories, diff, gold, slice_name, kind):
    return {
        "state": make_state(task, memories, diff),
        "questions": QUESTIONS,
        "gold": gold,
        "split": split_of(task),
        "slice": slice_name,
        "kind": kind,
        "validated": "structural",
        "teacher": MODEL,
    }


def pair(task, chosen, rejected, dims):
    return {"task": task, "chosen": chosen, "rejected": rejected, "dimensions": dims, "teacher": MODEL}


def load_pool():
    pool = []
    for name in SOURCES:
        p = os.path.join(CACHE, name + ".jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if not r.get("correct"):
                continue
            task = clean_task(r.get("task"))
            diff = (r.get("diff") or "").strip()[:MAX_CHARS]
            if task and diff and "@@" in diff and not suspect(diff):
                pool.append((task, diff, name))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of tasks to generate")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "gen_deepseek"))
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--bugs", type=int, default=2)
    ap.add_argument("--scopes", type=int, default=1)
    ap.add_argument("--focus", default="", choices=list(FOCUS_INSTRUCTION) + ["answer", "memory", "benign"])
    args = ap.parse_args()
    random.seed(args.seed)
    global NBUGS, NSCOPES, FOCUS
    NBUGS, NSCOPES, FOCUS = args.bugs, args.scopes, args.focus
    os.makedirs(args.out, exist_ok=True)

    rows_path = os.path.join(args.out, "rows_teacher.jsonl")
    pairs_path = os.path.join(args.out, "pairs_teacher.jsonl")
    manifest_path = os.path.join(args.out, "manifest.json")

    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_key"):
                done.add(r["task_key"])
    print(f"resume: {len(done)} tasks already done", flush=True)

    pool = load_pool()
    random.shuffle(pool)
    batch, seen = [], set()
    for idx, (task, diff, name) in enumerate(pool):
        if args.shard_count > 1 and idx % args.shard_count != args.shard_index:
            continue
        k = task_key(task, diff)
        if k in done or k in seen:
            continue
        seen.add(k)
        batch.append((task, diff, name, k))
        if len(batch) >= args.n:
            break
    print(f"queue: {len(batch)} tasks, model={MODEL}, workers={args.workers}\n", flush=True)

    def work(item):
        task, diff, name, k = item
        if FOCUS == "answer":
            res = gen_answer_task(task, diff)
        elif FOCUS == "memory":
            res = gen_memory_task(task, diff)
        elif FOCUS == "benign":
            res = gen_benign_task(task, diff)
        else:
            res = gen_task(task, diff)
        res["task_key"], res["source"] = k, name
        return res

    counts = {"ok": 0, "bug": 0, "scope": 0, "memories": 0, "fail": 0, "answer": 0, "memory_violation": 0, "benign": 0}
    t0 = time.time()
    with open(rows_path, "a") as rf, open(pairs_path, "a") as pf, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, it): it for it in batch}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            item = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                counts["fail"] += 1
                print(f"  [{i}/{len(batch)}] FAIL {item[2]} {item[3]}: {str(exc)[:160]}", flush=True)
                continue
            if res.get("mode") == "answer":
                counts["ok"] += 1
                emitted = []
                for text, level in res.get("answers", []):
                    if not isinstance(text, str) or len(text.strip()) < 15:
                        continue
                    emitted.append(
                        {
                            "state": {"task": res["question"], "answer": text.strip()[:1200]},
                            "questions": {"answer_quality": ANSWER_QUALITY_Q},
                            "gold": {"answer_quality": score(level)},
                            "split": split_of(res["question"] or res["task"]),
                            "slice": "teacher-answer",
                            "kind": "answer",
                            "validated": "structural",
                            "teacher": MODEL,
                            "task_key": res["task_key"],
                            "source": res["source"],
                        }
                    )
                for row in emitted:
                    out = dict(row)
                    out["questions"] = json.dumps(row["questions"])
                    out["gold"] = json.dumps(row["gold"])
                    rf.write(json.dumps(out) + "\n")
                    counts["answer"] += 1
                rf.flush()
                if i % 10 == 0 or i == len(batch):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [{i}/{len(batch)}] tasks={counts['ok']} answer={counts['answer']} fail={counts['fail']} ({rate:.3f} task/s)", flush=True)
                continue
            if res.get("mode") == "memory":
                counts["ok"] += 1
                task, memories = res["task"], res["memories"]
                counts["memories"] += len(memories)
                emitted = [
                    emit(task, memories, res["diff"], {"requirements_met": noul(True), "no_unrelated_changes": noul(True), "bug_risk": score(0)}, "teacher-memory-positive", "positive")
                ]
                for v in res["violations"]:
                    sev = int(v.get("severity", 2) or 2)
                    sev = 2 if sev < 2 else (3 if sev > 3 else sev)
                    emitted.append(emit(task, memories, v["mutated_diff"], {"requirements_met": noul(sev < 3), "no_unrelated_changes": noul(True), "bug_risk": score(sev)}, "teacher-memory-violation", "bug"))
                    counts["memory_violation"] += 1
                for row in emitted:
                    row["task_key"], row["source"] = res["task_key"], res["source"]
                    out = dict(row)
                    out["questions"] = json.dumps(row["questions"])
                    out["gold"] = json.dumps(row["gold"])
                    rf.write(json.dumps(out) + "\n")
                rf.flush()
                if i % 10 == 0 or i == len(batch):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [{i}/{len(batch)}] tasks={counts['ok']} mem_viol={counts['memory_violation']} fail={counts['fail']} ({rate:.3f} task/s)", flush=True)
                continue
            if res.get("mode") == "benign":
                counts["ok"] += 1
                task = res["task"]
                digest = hashlib.sha1(str(res["task_key"]).encode()).hexdigest()
                canned = dict(BENIGN_MEMORIES[int(digest[:8], 16) % len(BENIGN_MEMORIES)])
                canned["id"] = digest[:8] + "b"
                emitted = []
                for change in res["changes"]:
                    for memory in ([], [canned]):
                        emitted.append(
                            emit(
                                task,
                                memory,
                                change["mutated_diff"],
                                {"requirements_met": noul(True), "no_unrelated_changes": noul(True), "bug_risk": score(0)},
                                "teacher-benign",
                                "positive",
                            )
                        )
                        counts["benign"] += 1
                for row in emitted:
                    row["task_key"], row["source"] = res["task_key"], res["source"]
                    out = dict(row)
                    out["questions"] = json.dumps(row["questions"])
                    out["gold"] = json.dumps(row["gold"])
                    rf.write(json.dumps(out) + "\n")
                rf.flush()
                if i % 10 == 0 or i == len(batch):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [{i}/{len(batch)}] tasks={counts['ok']} benign={counts['benign']} fail={counts['fail']} ({rate:.3f} task/s)", flush=True)
                continue
            task, diff, memories = res["task"], res["diff"], res["memories"]
            counts["memories"] += len(memories)
            counts["ok"] += 1
            emitted = [
                emit(task, memories, diff, {"requirements_met": noul(True), "no_unrelated_changes": noul(True), "bug_risk": score(0)}, "teacher-positive", "positive")
            ]
            for b in res["bugs"]:
                sev = int(b.get("severity", 2) or 2)
                sev = 2 if sev < 2 else (3 if sev > 3 else sev)
                emitted.append(emit(task, memories, b["mutated_diff"], {"requirements_met": noul(not bool(b.get("requirement_impact", False))), "no_unrelated_changes": noul(True), "bug_risk": score(sev)}, "teacher-bug", "bug"))
                pf.write(json.dumps(pair(task, diff, b["mutated_diff"], {"approach": True, "minimal": True, "defect": str(b.get("bug_class", ""))})) + "\n")
                counts["bug"] += 1
            for s in res["scopes"]:
                risk = s.get("risk", 0)
                risk = int(risk) if isinstance(risk, (int, str)) and str(risk).isdigit() else 0
                emitted.append(emit(task, memories, s["mutated_diff"], {"requirements_met": noul(True), "no_unrelated_changes": noul(False), "bug_risk": score(min(risk, 1))}, "teacher-scope", "scope"))
                pf.write(json.dumps(pair(task, diff, s["mutated_diff"], {"approach": True, "minimal": False, "defect": ""})) + "\n")
                counts["scope"] += 1
            for row in emitted:
                row["task_key"], row["source"] = res["task_key"], res["source"]
                out = dict(row)
                out["questions"] = json.dumps(row["questions"])
                out["gold"] = json.dumps(row["gold"])
                rf.write(json.dumps(out) + "\n")
            rf.flush()
            pf.flush()
            if i % 10 == 0 or i == len(batch):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [{i}/{len(batch)}] tasks={counts['ok']} bug={counts['bug']} scope={counts['scope']} fail={counts['fail']} ({rate:.3f} task/s)", flush=True)

    manifest = {
        "model": MODEL,
        "seed": args.seed,
        "requested": args.n,
        "completed_tasks": counts["ok"],
        "failed_tasks": counts["fail"],
        "rows": {"positive": counts["ok"], "bug": counts["bug"], "scope": counts["scope"], "answer": counts["answer"], "memory_violation": counts["memory_violation"], "benign": counts["benign"]},
        "invariants": counts["memories"],
        "validated": "structural (teacher labels; not execution-verified)",
        "rows_path": rows_path,
        "pairs_path": pairs_path,
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
