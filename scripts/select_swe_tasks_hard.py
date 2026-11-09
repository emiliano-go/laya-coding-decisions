"""Pick hard-but-scorable instances: moderate FAIL_TO_PASS, bounded P2P, large repos."""
import json
import os

from datasets import load_dataset

OUT = os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "swe-smith", "hard.json")
TARGET = 26
NON_PYTHON = {
    "vbauerster__mpb", "prest__prest", "RoaringBitmap__roaring", "mgechev__revive",
    "spf13__cobra", "hamba__avro", "labstack__echo", "hjson__hjson-go", "ariga__atlas",
}
SMALL_REPOS = {
    "oauthlib__oauthlib", "pytest-dev__iniconfig", "jd__tenacity", "cknd__stackprinter",
    "luozhouyang__python-string-similarity", "madzak__python-json-logger", "seatgeek__thefuzz",
}
LARGE_HINT = {
    "pandas-dev__pandas", "pydantic__pydantic", "scrapy__scrapy", "tornadoweb__tornado",
    "encode__starlette", "sqlfluff__sqlfluff", "conan-io__conan", "pygments__pygments",
    "joke2k__faker", "pyparsing__pyparsing", "scanny__python-pptx", "Project-MONAI__MONAI",
    "facebookresearch__hydra", "pydicom__pydicom", "python-openxml__python-docx",
    "lepture__mistune", "kayak__pypika", "arrow-py__arrow", "pylint-dev__astroid",
    "python-jsonschema__jsonschema", "kurtmckee__feedparser", "bottlepy__bottle",
    "mahmoud__boltons", "agronholm__exceptiongroup", "PyCQA__flake8",
}

best = {}
ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
for row in ds:
    f2p, p2p = len(row["FAIL_TO_PASS"]), len(row["PASS_TO_PASS"])
    if not (3 <= f2p <= 15 and 50 <= p2p <= 400):
        continue
    repo = row["repo"]
    base = repo.split(".")[0].replace("swesmith/", "")
    if base in SMALL_REPOS or base in NON_PYTHON:
        continue
    rank = (base in LARGE_HINT, p2p, f2p)
    if repo not in best or rank > best[repo][0]:
        best[repo] = (
            rank,
            {
                "instance_id": row["instance_id"],
                "repo": repo,
                "image_name": row["image_name"],
                "problem_statement": row["problem_statement"],
                "FAIL_TO_PASS": list(row["FAIL_TO_PASS"]),
                "PASS_TO_PASS": list(row["PASS_TO_PASS"]),
            },
        )

ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)[:TARGET]
tasks = [item[1] for item in ranked]
with open(OUT, "w") as f:
    json.dump(tasks, f, indent=2)
print(f"wrote {len(tasks)} tasks to {OUT}")
for t in tasks:
    print(f"  f2p={len(t['FAIL_TO_PASS']):2d} p2p={len(t['PASS_TO_PASS']):3d}  {t['instance_id']}")
