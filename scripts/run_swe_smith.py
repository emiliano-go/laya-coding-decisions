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


def agent_run(repo, prompt, timeout_min):
    env = {**os.environ, "PWD": repo}
    try:
        proc = subprocess.run(
            ["bun", "run", ENTRY, "run", "--auto", "--model", AGENT, prompt],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_min * 60,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        return 124


def score(image, repo, task, p2p_sample):
    ids = list(task["FAIL_TO_PASS"]) + list(task["PASS_TO_PASS"][:p2p_sample])
    quoted = " ".join(f"'{i}'" for i in ids)
    proc = run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{repo}:/testbed",
            image, "bash", "-lc",
            f"cd /testbed && python -m pytest -q {quoted}",
        ],
        timeout=900,
    )
    return proc.returncode == 0, proc.stdout[-1500:], proc.stderr[-800:]


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
    ap.add_argument("--conditions", default="off,veto-turn,full-turn")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--timeout-min", type=int, default=15)
    ap.add_argument("--p2p-sample", type=int, default=8)
    ap.add_argument("--remove-images", action="store_true", help="docker rmi each image after its conditions")
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
        if not os.path.exists(os.path.join(base, "repo")):
            export_testbed(image, os.path.join(base, "repo"))
            sanitize(os.path.join(base, "repo"))
            task["oracle"] = hide_oracle(os.path.join(base, "repo"), task, os.path.join(base, "oracle"))
            with open(os.path.join(base, "task.json"), "w") as f:
                json.dump(task, f, indent=2)
        else:
            task = json.load(open(os.path.join(base, "task.json")))

        if unsupported(os.path.join(base, "repo")):
            for cond in conditions:
                if (iid, cond) in done:
                    continue
                row = {
                    "instance_id": iid,
                    "repo": task["repo"],
                    "condition": cond,
                    "passed": None,
                    "skipped": "unsupported-language",
                    "agent_exit": None,
                    "seconds": 0,
                    "coverage": {"step": 0, "mutation": 0, "answer": 0},
                    "score_tail": "",
                    "score_err": "",
                }
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {cond}: skipped (unsupported language)", flush=True)
            continue

        for cond in conditions:
            if (iid, cond) in done:
                print(f"  {cond}: already done", flush=True)
                continue
            repo = os.path.join(base, f"repo-{cond}")
            copy_repo(os.path.join(base, "repo"), repo)
            decision_config(cond, repo)
            start = time.time()
            code = agent_run(repo, task["problem_statement"], args.timeout_min)
            restore_oracle(repo, os.path.join(base, "oracle"), task.get("oracle", []))
            ok, out, err = score(image, repo, task, args.p2p_sample)
            cov = coverage(repo)
            row = {
                "instance_id": iid,
                "repo": task["repo"],
                "condition": cond,
                "passed": ok,
                "agent_exit": code,
                "seconds": round(time.time() - start, 1),
                "coverage": cov,
                "score_tail": out,
                "score_err": err,
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {cond}: passed={ok} agent_exit={code} step={cov['step']} term={cov['mutation'] + cov['answer']} ({row['seconds']}s)", flush=True)
        if args.remove_images:
            run(["docker", "rmi", image])
            print(f"  removed image {image}", flush=True)


if __name__ == "__main__":
    main()
