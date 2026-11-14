"""Run an agent on SWE-smith instances with decision-engine conditions.

Pipeline per instance:
  image -> /testbed copy -> sanitize -> hide FAIL_TO_PASS tests -> per condition:
  copy -> agent run -> restore tests -> score in container (network isolated).

Usage:
  python run_swe_smith.py --tasks /data/laya-ft/swe-smith/tasks.json \
    --root /data/laya-ft/swe-smith --conditions off,veto-turn,full-turn \
    --limit 20 --timeout-min 15
"""
import argparse
import json
import os
import shutil
import subprocess
import time

ENCODER = os.environ.get("ENCODER_ROOT", "/path/to/encoder")
ENTRY = f"{ENCODER}/packages/encoder/src/index.ts"
QUESTIONS = f"{ENCODER}/.encoder/decision.questions.json"
AGENT = os.environ.get("AGENT_MODEL", "opencode-go/deepseek-v4.1-flash")
BASE_URL = os.environ.get("DECISION_BASE_URL", "http://127.0.0.1:8765")

CONDITIONS = {
    "off": {"enabled": False},
    "veto-turn": {"enabled": True, "stepChecks": "veto", "stepCadence": "turn"},
    "veto-tool": {"enabled": True, "stepChecks": "veto", "stepCadence": "tool"},
    "veto-both": {"enabled": True, "stepChecks": "veto", "stepCadence": "both"},
    "full-turn": {"enabled": True, "stepChecks": "full", "stepCadence": "turn"},
    "full-tool": {"enabled": True, "stepChecks": "full", "stepCadence": "tool"},
    "full-both": {"enabled": True, "stepChecks": "full", "stepCadence": "both"},
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def image_present(image):
    return run(["docker", "image", "inspect", image]).returncode == 0


def pull(image):
    if image_present(image):
        return
    for attempt in range(3):
        proc = run(["docker", "pull", image], timeout=1800)
        if proc.returncode == 0:
            return
        print(f"  pull retry {attempt + 1}: {proc.stderr[-200:]}", flush=True)
        time.sleep(10)
    raise RuntimeError(f"could not pull {image}")


def export_testbed(image, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    uid, gid = os.getuid(), os.getgid()
    proc = run(
        ["docker", "run", "--rm", "--user", f"{uid}:{gid}", "-v", f"{dest}:/out", image, "bash", "-lc", "cp -a /testbed/. /out/"],
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"export failed: {proc.stderr[-300:]}")


def apply_bug_patch(repo, task):
    """SWE-smith's `patch` is the bug-injection diff, not the fix."""
    patch = task.get("patch") or ""
    if not patch.strip():
        return True, ""
    proc = run(["git", "apply", "--whitespace=nowarn", "-p1", "-"], cwd=repo, input=patch)
    return proc.returncode == 0, proc.stderr[-300:]


def unsupported(repo):
    """Non-Python images cannot be scored by `python -m pytest`; skip them."""
    return any(
        os.path.exists(os.path.join(repo, marker))
        for marker in ("go.mod", "Cargo.toml", "build.gradle", "pom.xml", "package.json")
    )


def copy_repo(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if run(["cp", "-a", "--reflink=auto", src, dst]).returncode != 0:
        shutil.copytree(src, dst, symlinks=True)


def sanitize(repo):
    for name in [".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox"]:
        shutil.rmtree(os.path.join(repo, name), ignore_errors=True)
    for root, dirs, files in os.walk(repo):
        for d in list(dirs):
            if d in ("__pycache__", ".pytest_cache", ".mypy_cache"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        for f in files:
            if f.endswith((".pyc", ".pyo", ".orig", ".rej", ".patch", ".log")):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "add", "-A"], cwd=repo)
    run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=agent", "commit", "-qm", "initial"],
        cwd=repo,
    )


def hide_oracle(repo, task, oracle_dir):
    os.makedirs(oracle_dir, exist_ok=True)
    moved = []
    files = sorted({tid.split("::", 1)[0] for tid in task["FAIL_TO_PASS"]})
    for rel in files:
        src = os.path.join(repo, rel)
        if os.path.isfile(src):
            dst = os.path.join(oracle_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(rel)
    return moved


def restore_oracle(repo, oracle_dir, moved):
    for rel in moved:
        src = os.path.join(oracle_dir, rel)
        dst = os.path.join(repo, rel)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def decision_config(cond, repo):
    base = {
        "enabled": True,
        "baseUrl": BASE_URL,
        "confidenceFloor": 0,
        "passThreshold": 0.65,
        "maxFeedbackRetries": 1,
        "captureState": True,
        **CONDITIONS[cond],
    }
    os.makedirs(os.path.join(repo, ".encoder"), exist_ok=True)
    with open(os.path.join(repo, ".encoder", "encoder.json"), "w") as f:
        json.dump({"snapshot": False, "decision": base}, f)
    shutil.copy2(QUESTIONS, os.path.join(repo, ".encoder", "decision.questions.json"))


def agent_problem(log_path, code):
    try:
        text = open(log_path, errors="replace").read()
    except OSError:
        return "agent-no-log"
    if code == 0:
        return None if text.strip() else "agent-no-output"
    if "usage limit exceeded" in text:
        return "provider-quota"
    if len(text.strip()) < 60:
        return "agent-no-output"
    return None


def agent_run(repo, prompt, timeout_min, log_path, agent=AGENT):
    env = {**os.environ, "PWD": repo}
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                ["bun", "run", ENTRY, "run", "--auto", "--print-logs", "--log-level", "ERROR", "--model", agent, prompt],
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_min * 60,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            return 124


def run_pytest(image, repo, ids, timeout=900):
    quoted = " ".join(f"'{i}'" for i in ids)
    proc = run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{repo}:/testbed",
            image, "bash", "-lc",
            f"cd /testbed && python -m pytest -q -p no:cacheprovider {quoted}",
        ],
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout[-6000:], proc.stderr[-800:]


def score(image, repo, task, p2p_sample):
    ids = list(task["FAIL_TO_PASS"]) + list(task["PASS_TO_PASS"][:p2p_sample])
    ok, out, err = run_pytest(image, repo, ids)
    return ok, out[-1500:], err


def describe_failure(output):
    """Turn the baseline pytest failure into a readable issue for the agent."""
    lines = [line for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-40:])
    return "Fix the failing tests.\n\nBaseline failure output:\n" + tail[-4000:]


def write_row(results_path, iid, task, cond, passed, skipped=None, agent_exit=None, seconds=0, cov=None, out="", err="", invalid=None, log_path=None):
    row = {
        "instance_id": iid,
        "repo": task.get("repo"),
        "condition": cond,
        "passed": passed,
        "agent_exit": agent_exit,
        "seconds": seconds,
        "coverage": cov or {"step": 0, "mutation": 0, "answer": 0},
        "score_tail": out,
        "score_err": err,
        "invalid": invalid,
        "agent_log": log_path,
    }
    if skipped:
        row["skipped"] = skipped
    with open(results_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def coverage(repo):
    path = os.path.join(repo, ".encoder", "decision-log.jsonl")
    counts = {"step": 0, "mutation": 0, "answer": 0}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = entry.get("kind")
            if kind in counts:
                counts[kind] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "swe-smith", "tasks.json"))
    ap.add_argument("--root", default=os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "swe-smith"))
    ap.add_argument("--conditions", default="off,veto-turn")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--timeout-min", type=int, default=8)
    ap.add_argument("--p2p-sample", type=int, default=8)
    ap.add_argument("--remove-images", action="store_true", help="docker rmi each image after its conditions")
    ap.add_argument("--agent", default=AGENT, help="provider/model for the agent")
    ap.add_argument("--baseline-only", action="store_true", help="validate bug patch + failing F2P, do not run the agent")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))[args.skip : args.skip + args.limit]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    os.makedirs(args.root, exist_ok=True)
    results_path = os.path.join(args.root, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            try:
                r = json.loads(line)
                done.add((r["instance_id"], r["condition"]))
            except json.JSONDecodeError:
                pass

    for index, task in enumerate(tasks, 1):
        iid = task["instance_id"]
        base = os.path.join(args.root, iid)
        image = task["image_name"]
        print(f"\n[{index}/{len(tasks)}] {iid}", flush=True)
        pull(image)
        task_path = os.path.join(base, "task.json")
        task = json.load(open(task_path)) if os.path.exists(task_path) else task
        if (
            not os.path.exists(task_path)
            or task.get("baseline_ok") is None
            or not task.get("patch")
            or not task.get("patch_applied")
        ):
            shutil.rmtree(base, ignore_errors=True)
            export_testbed(image, os.path.join(base, "repo"))
            applied, apply_err = apply_bug_patch(os.path.join(base, "repo"), task)
            task["patch_applied"] = applied
            if not applied:
                print(f"  bug patch failed to apply: {apply_err}", flush=True)
            sanitize(os.path.join(base, "repo"))
            if unsupported(os.path.join(base, "repo")):
                task["baseline_ok"] = False
                task["unsupported"] = True
            elif not applied:
                task["baseline_ok"] = False
            else:
                ok, f2p_out, _ = run_pytest(image, os.path.join(base, "repo"), task["FAIL_TO_PASS"])
                task["baseline_ok"] = not ok
                task["baseline_tail"] = f2p_out[-1200:]
                if ok:
                    print("  baseline passes F2P; instance is invalid", flush=True)
                elif not (task.get("problem_statement") or "").strip():
                    task["problem_statement"] = describe_failure(f2p_out)
                    print("  synthesized problem statement from baseline output", flush=True)
            if task.get("baseline_ok"):
                task["oracle"] = hide_oracle(os.path.join(base, "repo"), task, os.path.join(base, "oracle"))
            with open(task_path, "w") as f:
                json.dump(task, f, indent=2)
            if not task.get("baseline_ok") and not task.get("unsupported"):
                reason = "patch-apply-failed" if not applied else "baseline-passes"
                for cond in conditions:
                    if (iid, cond) in done:
                        continue
                    write_row(results_path, iid, task, cond, None, reason)
                if args.remove_images:
                    run(["docker", "rmi", image])
                continue
            if args.baseline_only:
                print(f"  baseline fails F2P; valid (f2p={len(task['FAIL_TO_PASS'])})", flush=True)
                if args.remove_images:
                    run(["docker", "rmi", image])
                continue
        else:
            task = json.load(open(task_path))

        if task.get("unsupported") or unsupported(os.path.join(base, "repo")):
            for cond in conditions:
                if (iid, cond) in done:
                    continue
                write_row(results_path, iid, task, cond, None, "unsupported-language")
                print(f"  {cond}: skipped (unsupported language)", flush=True)
            continue

        for cond in conditions:
            if (iid, cond) in done:
                print(f"  {cond}: already done", flush=True)
                continue
            repo = os.path.join(base, f"repo-{cond}")
            copy_repo(os.path.join(base, "repo"), repo)
            decision_config(cond, repo)
            log_path = os.path.join(base, f"agent-{cond}.log")
            start = time.time()
            code = agent_run(repo, task["problem_statement"], args.timeout_min, log_path, args.agent)
            problem = agent_problem(log_path, code)
            if problem == "provider-quota":
                write_row(results_path, iid, task, cond, None, agent_exit=code, seconds=round(time.time() - start, 1), invalid=problem, log_path=log_path)
                print(f"  {cond}: {problem}; stopping run", flush=True)
                raise SystemExit(2)
            restore_oracle(repo, os.path.join(base, "oracle"), task.get("oracle", []))
            ok, out, err = score(image, repo, task, args.p2p_sample)
            cov = coverage(repo)
            seconds = round(time.time() - start, 1)
            invalid = None
            if cond != "off" and sum(cov.values()) == 0:
                invalid = "no-decision-eval"
            elif code not in (0, 124):
                invalid = f"agent-exit-{code}"
            write_row(results_path, iid, task, cond, ok, agent_exit=code, seconds=seconds, cov=cov, out=out, err=err, invalid=invalid, log_path=log_path)
            print(f"  {cond}: passed={ok} agent_exit={code} step={cov['step']} term={cov['mutation'] + cov['answer']} invalid={invalid} ({seconds}s)", flush=True)
            if invalid and os.path.exists(log_path):
                print("  agent log tail:", "".join(open(log_path, errors="replace").readlines()[-6:]).strip()[:400], flush=True)
        if args.remove_images:
            run(["docker", "rmi", image])
            print(f"  removed image {image}", flush=True)


if __name__ == "__main__":
    main()
