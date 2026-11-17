"""Unit checks for the frozen extraction and shortlist rules.

Stdlib only, no server, no repo needed:

    python scripts/test_localize_candidates.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import localize_candidates as lc  # noqa: E402

SAMPLE = '''\
"""Module docstring."""

import os


class Base:
    """A base."""

    def tell(self) -> int:
        """Return the position."""
        return self._tell

    @property
    def length(self):
        return self._tell


class Derived(Base):
    @staticmethod
    async def fetch(self, url, *, timeout=5):
        # a comment inside a wrapped signature
        return url


def overloaded(value: int):
    ...


def overloaded(value: str):  # type: ignore
    return value


def outer():
    def inner():
        return 1

    return inner
'''

PATCHED = '''\
 line one
+from __future__ import annotations
'''


def test_extract_structure() -> None:
    found = {candidate["id"]: candidate for candidate in lc.extract_candidates("pkg/mod.py", SAMPLE)}
    for ident in (
        "pkg/mod.py:Base.tell",
        "pkg/mod.py:Base.length",
        "pkg/mod.py:Derived.fetch",
        "pkg/mod.py:overloaded",
        "pkg/mod.py:outer",
        "pkg/mod.py:outer.inner",
        "pkg/mod.py::<module>",
    ):
        assert ident in found, (ident, sorted(found))
    assert found["pkg/mod.py:Base.tell"]["args"] == "self"
    assert found["pkg/mod.py:Base.tell"]["text"].startswith("pkg/mod.py:Base.tell()")
    assert found["pkg/mod.py:Derived.fetch"]["args"] == "self, url, *, timeout=5"
    assert found["pkg/mod.py:Derived.fetch"]["text"].startswith("pkg/mod.py:Derived.fetch(url")
    assert "comment" not in found["pkg/mod.py:Derived.fetch"]["args"]
    assert "position" in found["pkg/mod.py:Base.tell"]["text"]
    assert found["pkg/mod.py:outer.inner"]["qualname"] == "outer.inner"
    # An overload set keeps every definition addressable.
    overloads = [ident for ident in found if ident.startswith("pkg/mod.py:overloaded")]
    assert len(overloads) == 2, sorted(overloads)
    assert "pkg/mod.py:overloaded" in overloads


def test_spans() -> None:
    found = lc.extract_candidates("pkg/mod.py", SAMPLE)
    tell = next(candidate for candidate in found if candidate["id"] == "pkg/mod.py:Base.tell")
    assert tell["line"] == 9, tell
    lines = SAMPLE.splitlines()
    body = "\n".join(lines[tell["line"] - 1:tell["end"]])
    assert "return self._tell" in body, (tell["line"], tell["end"], body)
    derived = next(candidate for candidate in found if candidate["id"] == "pkg/mod.py:Derived.fetch")
    assert "return url" in "\n".join(lines[derived["line"] - 1:derived["end"]])


def test_frames() -> None:
    failure = (
        "tests/test_x.py:12: in test_parse\n"
        "    assert value == 3\n"
        '  File "/testbed/pkg/mod.py", line 42, in tell\n'
        "    return self._tell + 1\n"
        "pkg/mod.py:42: in tell\n"
        "E   AssertionError\n"
        "FAILED tests/test_x.py::test_parse - AssertionError\n"
        "FAILED tests/test_y.py::test_other\n"
    )
    frames = lc.parse_frames(failure, "/data/repo")
    assert frames[0] == {"file": "tests/test_x.py", "line": 12, "func": "test_parse"}, frames
    assert {"file": "pkg/mod.py", "line": 42, "func": "tell"} in frames, frames
    # Consecutive repeats are collapsed.
    assert len(frames) == 2, frames
    assert lc.parse_failed_tests(failure) == ["tests/test_x.py::test_parse", "tests/test_y.py::test_other"]
    assert not [line for line in lc.query_text(failure).splitlines() if set(line) <= {"."}]


def test_paths() -> None:
    assert lc.is_test_path("tests/test_x.py")
    assert lc.is_test_path("pkg/test_y.py")
    assert lc.is_test_path("pkg/conftest.py")
    assert not lc.is_test_path("pkg/testing_helpers.py")
    assert lc.strip_root('/testbed/pkg/mod.py', "/data/repo") == "pkg/mod.py"
    assert lc.strip_root("/data/repo/pkg/mod.py", "/data/repo") == "pkg/mod.py"


def test_ranking() -> None:
    candidates = [
        {"id": "a:alpha", "path": "a.py", "qualname": "alpha", "kind": "function", "line": 1, "end": 20, "args": "", "text": "a.py:alpha() — parse query"},
        {"id": "a:beta", "path": "a.py", "qualname": "beta", "kind": "function", "line": 22, "end": 40, "args": "", "text": "a.py:beta() — unrelated"},
        {"id": "b:gamma", "path": "b.py", "qualname": "gamma", "kind": "function", "line": 1, "end": 10, "args": "", "text": "b.py:gamma() — parse query"},
    ]
    failure = "a.py:15: in alpha\n    raise ValueError\nFAILED tests/test_a.py::test_alpha\n"
    ranked = lc.rank_candidates(candidates, failure, "query parsing", 2)
    assert ranked["stratum"] == "frame"
    assert ranked["ranked"][0]["id"] == "a:alpha", ranked["ranked"]
    # Assertion-only output has no source frame, so nothing gets the frame bonus.
    only_assert = "tests/test_a.py:9: in test_alpha\nE   assert 1 == 2\n"
    assert lc.rank_candidates(candidates, only_assert, "query parsing", 2)["stratum"] == "assert-only"
    assert lc.rank_candidates(candidates, "", "", 2)["stratum"] == "none"
    # Scores are bounded by the weights: no raw BM25 mega-scores.
    assert len(lc.bm25(["a"], [["a"], ["b"]])) == 2


def test_choice_body() -> None:
    candidates = lc.extract_candidates("pkg/mod.py", SAMPLE)[:3]
    body = lc.build_choice(candidates, "x" * 900, "y" * 5000, tail=1000)
    assert len(body["state"]["task"]) == lc.MAX_TASK
    assert len(body["state"]["changes"][0]["output"]) == 1000
    assert list(body["questions"]["loc"]["criteria"]) == [candidate["id"] for candidate in candidates]
    assert body["questions"]["loc"]["type"] == "choice"


def test_walk_repo_excludes() -> None:
    with tempfile.TemporaryDirectory() as root:
        def write(rel: str, text: str = "x = 1\n") -> None:
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(text)

        write("pkg/mod.py", "def f():\n    return 1\n")
        write("tests/test_mod.py")
        write("pkg/test_other.py")
        write(".venv/lib/site.py")
        write("node_modules/x/index.py")
        write("docs/conf.py")
        write("pkg/notes.txt")
        found = lc.walk_repo(root)
        paths = sorted(item["path"] for item in found)
        assert paths == ["pkg/mod.py"], paths


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
