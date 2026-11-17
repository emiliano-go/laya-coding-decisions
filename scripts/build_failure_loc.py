"""Materialize SWE-smith instances as failure-localization cases.

Per instance: export the testbed image, apply the injected bug patch, sanitize
(no `.git` history left behind), run the hidden FAIL_TO_PASS tests to capture the
real failure output, move those test files out of the working tree, and write
the gold location plus the frozen candidate list.

No agent runs. Docker is used only to export the repo and run pytest offline.

    python scripts/build_failure_loc.py --tasks /data/laya-ft/swe-smith/hard.json \
      --root /data/laya-ft/failure-loc --limit 21 --resume

Each instance directory contains:

    task.json        original SWE-smith row (has `patch`; never shown to Laya)
    repo/            buggy working tree
    oracle/          the FAIL_TO_PASS test files, moved out of the tree
    failure.txt      raw pytest output at the buggy commit
    gold.json        {"files": {path: [added line numbers]}, "sites": [...]}
    candidates.json  frozen candidate list (see localize_candidates.py)
    status.json      ok/invalid + reason + counts
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import localize_candidates as lc  # noqa: E402
import run_swe_smith as harness  # noqa: E402
from localize_probe import patch_added_lines  # noqa: E402

ALLOW_PULL = True


def write_json(path: str, payload: object) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def classify(task: dict, repo: str) -> str | None:
    """Reason this instance cannot be localized, or None when it is usable."""
    if harness.unsupported(repo):
        return "unsupported-language"
    if not (task.get("patch") or "").strip():
        return "no-patch"
    if not task.get("FAIL_TO_PASS"):
        return "no-fail-to-pass"
    return None


def gold_of(task: dict, repo: str, candidates: list[dict]) -> dict | None:
    """Added-line map plus the candidate ids those lines fall inside.

    Gold is resolved against the frozen candidate spans rather than an AST so a
    gold id is always present in `candidates.json` (overloads and nested
    functions otherwise produce ids the extractor never emits).
    """
    added = patch_added_lines(task.get("patch") or "")
    files = {
        path: sorted(lines)
        for path, lines in added.items()
        if lines and not lc.is_test_path(path) and os.path.isfile(os.path.join(repo, path))
    }
    if not files:
        return None
    sites: list[str] = []
    for path, lines in files.items():
        same_file = [candidate for candidate in candidates if candidate["path"] == path]
        for line in lines:
            inside = [
                candidate for candidate in same_file
                if candidate["line"] <= line <= candidate["end"]
            ]
            if not inside:
                continue
            # Smallest span wins: a nested function beats its enclosing class body.
            inside.sort(key=lambda candidate: (candidate["end"] - candidate["line"], candidate["line"]))
            if inside[0]["id"] not in sites:
                sites.append(inside[0]["id"])
    if not sites:
        return None
    return {"files": files, "sites": sites, "primary": sites[0]}


def build_one(task: dict, root: str, tail: int) -> dict:
    iid = task["instance_id"]
    base = os.path.join(root, iid)
    repo = os.path.join(base, "repo")
    started = time.time()
    status = {"instance_id": iid, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}

    image = task.get("image_name", "")
    if not image:
        return {**status, "ok": False, "reason": "no-image"}
    if harness.image_present(image):
        pass
    elif not ALLOW_PULL:
        return {**status, "ok": False, "reason": "image-missing"}
    else:
        try:
            harness.pull(image)
        except Exception as error:  # noqa: BLE001 - docker failures are data
            return {**status, "ok": False, "reason": f"image-unavailable: {error}"}

    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    harness.export_testbed(image, repo)
    applied, apply_err = harness.apply_bug_patch(repo, task)
    if not applied:
        return {**status, "ok": False, "reason": f"patch-apply-failed: {apply_err}"}
    harness.sanitize(repo)

    reason = classify(task, repo)
    if reason is not None:
        return {**status, "ok": False, "reason": reason}

    ok, stdout, stderr = harness.run_pytest(image, repo, list(task["FAIL_TO_PASS"]), timeout=1200)
    failure = (stdout or "") + (("\n" + stderr) if stderr else "")
    if ok:
        return {**status, "ok": False, "reason": "baseline-passes"}

    with open(os.path.join(base, "failure.txt"), "w") as handle:
        handle.write(failure[-tail:])

    moved = harness.hide_oracle(repo, task, os.path.join(base, "oracle"))
    candidates: list[dict] = []
    for item in lc.walk_repo(repo):
        candidates.extend(lc.extract_candidates(item["path"], item["text"]))

    gold = gold_of(task, repo, candidates)
    write_json(os.path.join(base, "task.json"), task)
    write_json(os.path.join(base, "candidates.json"), {"count": len(candidates), "candidates": candidates})
    write_json(os.path.join(base, "status.json"), {
        **status,
        "ok": gold is not None,
        "reason": None if gold is not None else "no-gold",
        "moved_tests": moved,
        "fail_to_pass": len(task["FAIL_TO_PASS"]),
        "candidates": len(candidates),
        "sites": len(gold["sites"]) if gold else 0,
        "seconds": round(time.time() - started, 1),
    })
    if gold is not None:
        write_json(os.path.join(base, "gold.json"), gold)
    return read_json(os.path.join(base, "status.json")) or status


def refresh_one(root: str, iid: str) -> dict | None:
    """Recompute candidates and gold from an already-materialized repo.

    Needed whenever the frozen extraction rules change: gold ids are candidate
    ids, so a rule change invalidates them without touching the repo or the
    captured failure output.
    """
    base = os.path.join(root, iid)
    repo = os.path.join(base, "repo")
    task = read_json(os.path.join(base, "task.json"))
    status = read_json(os.path.join(base, "status.json"))
    if task is None or status is None or not os.path.isdir(repo):
        return None
    if not status.get("ok"):
        return status
    candidates: list[dict] = []
    for item in lc.walk_repo(repo):
        candidates.extend(lc.extract_candidates(item["path"], item["text"]))
    gold = gold_of(task, repo, candidates)
    write_json(os.path.join(base, "candidates.json"), {"count": len(candidates), "candidates": candidates})
    if gold is None:
        status = {**status, "ok": False, "reason": "no-gold", "candidates": len(candidates), "sites": 0}
        write_json(os.path.join(base, "status.json"), status)
        return status
    write_json(os.path.join(base, "gold.json"), gold)
    status = {**status, "candidates": len(candidates), "sites": len(gold["sites"])}
    write_json(os.path.join(base, "status.json"), status)
    return status


def run_pytest_full(image: str, repo: str, ids: list[str], timeout: int = 1800, no_cache: bool = True) -> tuple[bool, str, str]:
    """pytest without the harness's 6 kB stdout truncation.

    `run_swe_smith.run_pytest` returns only the tail, which can be a plugin's
    teardown traceback rather than the test failure. The full text is focused
    with `localize_candidates.focus_failure` before it is stored.
    """
    quoted = " ".join(f"'{ident}'" for ident in ids)
    proc = harness.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{repo}:/testbed",
            image, "bash", "-lc",
            f"cd /testbed && /opt/miniconda3/envs/testbed/bin/python -m pytest -q -rA "
            + ("-p no:cacheprovider " if no_cache else "")
            + quoted,
        ],
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout or "", proc.stderr or ""


def recapture_one(root: str, iid: str, tail: int) -> dict | None:
    """Re-run pytest against a materialized repo and rewrite `failure.txt`.

    Restores the hidden test files, runs the suite, then hides them again so the
    repo state stays identical to what the localization sees.
    """
    base = os.path.join(root, iid)
    repo = os.path.join(base, "repo")
    oracle = os.path.join(base, "oracle")
    task = read_json(os.path.join(base, "task.json"))
    status = read_json(os.path.join(base, "status.json"))
    if task is None or status is None or not status.get("ok") or not os.path.isdir(repo):
        return None
    moved = status.get("moved_tests") or []
    harness.restore_oracle(repo, oracle, moved)
    image = task.get("image_name", "")
    ok, stdout, stderr = run_pytest_full(image, repo, list(task["FAIL_TO_PASS"]))
    raw = stdout + (("\n" + stderr) if stderr else "")
    if "INTERNALERROR" in raw:
        # A plugin needs the cacheprovider; retry without disabling it rather
        # than discarding an otherwise valid instance.
        ok, stdout, stderr = run_pytest_full(image, repo, list(task["FAIL_TO_PASS"]), no_cache=False)
        raw = stdout + (("\n" + stderr) if stderr else "")
    if ok:
        harness.hide_oracle(repo, task, oracle)
        return {**status, "ok": False, "reason": "recapture-baseline-passes"}
    if "INTERNALERROR" in raw:
        harness.hide_oracle(repo, task, oracle)
        return {**status, "ok": False, "reason": "pytest-internal-error"}
    # Store the cleaned raw output: frames are parsed from it, and the model
    # view is focused at query time (`localize_candidates.focus_failure`).
    text = lc.clean_stdout(raw)[:60000]
    with open(os.path.join(base, "failure.txt"), "w") as handle:
        handle.write(text)
    harness.hide_oracle(repo, task, oracle)
    status = {**status, "failure_chars": len(text), "frames": len(lc.parse_frames(text, repo))}
    write_json(os.path.join(base, "status.json"), status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="/data/laya-ft/swe-smith/hard.json")
    parser.add_argument("--root", default="/data/laya-ft/failure-loc")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--tail", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pull", action="store_true", help="skip instances whose image is not local")
    parser.add_argument("--refresh", action="store_true", help="recompute candidates/gold from materialized repos")
    parser.add_argument("--recapture", action="store_true", help="re-run pytest and rewrite failure.txt")
    parser.add_argument("--only", default="", help="comma-separated instance ids")
    args = parser.parse_args()
    global ALLOW_PULL
    ALLOW_PULL = not args.no_pull

    tasks = json.load(open(args.tasks))
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        tasks = [task for task in tasks if task["instance_id"] in wanted]
    tasks = tasks[args.skip:]
    if args.limit:
        tasks = tasks[:args.limit]

    os.makedirs(args.root, exist_ok=True)
    if args.refresh:
        rows = []
        for name in sorted(os.listdir(args.root)):
            if not os.path.isdir(os.path.join(args.root, name)):
                continue
            refreshed = refresh_one(args.root, name)
            if refreshed is None:
                continue
            rows.append(refreshed)
            print(f"{name[:56]:56s} ok={refreshed.get('ok')} cands={refreshed.get('candidates')} sites={refreshed.get('sites')}", flush=True)
        usable = [row for row in rows if row.get("ok")]
        print(json.dumps({"refreshed": len(rows), "usable": len(usable), "sites": sum(row.get("sites", 0) for row in usable)}, indent=1))
        return 0

    if args.recapture:
        rows = []
        for name in sorted(os.listdir(args.root)):
            if not os.path.isdir(os.path.join(args.root, name)):
                continue
            status = recapture_one(args.root, name, args.tail)
            if status is None:
                continue
            rows.append(status)
            print(f"{name[:56]:56s} ok={status.get('ok')} chars={status.get('failure_chars')} frames={status.get('frames')}", flush=True)
        print(json.dumps({"recaptured": len(rows)}, indent=1))
        return 0

    manifest = os.path.join(args.root, "manifest.jsonl")
    done = {}
    if args.resume and os.path.exists(manifest):
        for line in open(manifest):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["instance_id"]] = row
    else:
        open(manifest, "w").close()

    results = []
    for index, task in enumerate(tasks, 1):
        iid = task["instance_id"]
        if args.resume and done.get(iid, {}).get("ok"):
            print(f"[{index}/{len(tasks)}] {iid}: cached ok", flush=True)
            results.append(done[iid])
            continue
        print(f"[{index}/{len(tasks)}] {iid}", flush=True)
        try:
            status = build_one(task, args.root, args.tail)
        except Exception as error:  # noqa: BLE001 - keep the run alive
            status = {"instance_id": iid, "ok": False, "reason": f"error: {error}"}
        results.append(status)
        with open(manifest, "a") as handle:
            handle.write(json.dumps(status) + "\n")
        print(
            f"  ok={status.get('ok')} reason={status.get('reason')} "
            f"cands={status.get('candidates')} sites={status.get('sites')} "
            f"f2p={status.get('fail_to_pass')} ({status.get('seconds')}s)",
            flush=True,
        )

    usable = [row for row in results if row.get("ok")]
    print(json.dumps({
        "instances": len(results),
        "usable": len(usable),
        "sites": sum(row.get("sites", 0) for row in usable),
        "reasons": {
            reason: sum(1 for row in results if row.get("reason") == reason)
            for reason in sorted({row.get("reason") for row in results if row.get("reason")})
        },
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
