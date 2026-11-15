"""Sweep head-only merges and score each on the 33-case funnel.

  python scripts/merge_sweep.py --modes task,interp
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from merge_heads import merge

ENC = os.environ.get("ENCODER_ROOT", "/path/to/encoder")
PY = os.environ.get("PYTHON", sys.executable)
SERVER = os.path.join(ENC, "script", "laya-server.py")
PORT = 8766
OUT = os.environ.get("LAYACD_DATA", "/data/laya-ft") + "/merge-tmp"
QUESTIONS = os.environ.get("LAYACD_DATA", "/data/laya-ft") + "/questions3.json"


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def bench(tag: str) -> dict:
    env = {
        **os.environ,
        "DECISION_BASE_URL": f"http://127.0.0.1:{PORT}",
        "DECISION_QUESTIONS_FILE": QUESTIONS,
        "DECISION_PASS_THRESHOLD": "0.65",
        "DECISION_VETO_THRESHOLD": "0.5",
        "DECISION_BENCH_JSON": f"/data/laya-ft/bench-merge-{tag}.json",
    }
    proc = subprocess.run(["bun", "packages/encoder/script/decision-bench.ts"], cwd=ENC, env=env,
                          capture_output=True, text=True, timeout=900)
    text = proc.stdout + proc.stderr
    caught = re.search(r"slop caught:\s+(\d+)/(\d+)", text)
    false_pass = re.search(r"false pass:\s+(\d+)/(\d+)", text)
    false_alarm = re.search(r"false alarm:\s+(\d+)/(\d+)", text)
    return {
        "caught": int(caught.group(1)) if caught else None,
        "slop": int(caught.group(2)) if caught else None,
        "false_pass": int(false_pass.group(1)) if false_pass else None,
        "false_alarm": int(false_alarm.group(1)) if false_alarm else None,
        "good": int(false_alarm.group(2)) if false_alarm else None,
        "tail": text[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="/data/laya-ft/out-m5mem4-head")
    parser.add_argument("--b", default="/data/laya-ft/out-m6-head")
    parser.add_argument("--base", default="/data/laya-ft/out-multi")
    parser.add_argument("--modes", default="task,interp")
    parser.add_argument("--alphas", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--out", default="/data/laya-ft/merge-sweep.json")
    parser.add_argument("--only-scorer", action="store_true")
    args = parser.parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    results = []

    # Baseline m5mem4 once, then each merge.
    modes = args.modes.split(",")
    jobs = [("base", 0.0)] + [(mode, alpha) for mode in modes for alpha in alphas]
    for mode, alpha in jobs:
            tag = f"{mode}{'-scorer' if args.only_scorer else ''}-{alpha:.2f}"
            model = args.a
            if alpha > 0:
                merge(args.a, args.b, args.base, mode, alpha, OUT, args.only_scorer)
                model = OUT
            print(f"=== {tag} ({model})", flush=True)
            server = subprocess.Popen([PY, SERVER, "--port", str(PORT), "--model", model],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.time() + 90
                while time.time() < deadline and not healthy():
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}")
                    time.sleep(0.5)
                if not healthy():
                    raise RuntimeError("server did not become healthy")
                result = bench(tag)
            except Exception as error:
                result = {"error": str(error)}
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
            print(tag, result, flush=True)
            results.append({"tag": tag, "mode": mode, "alpha": alpha, **result})
    with open(args.out, "w") as sink:
        json.dump(results, sink, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
