"""Compile every Rust code block in haico's library articles.

An article is read by a model that is about to copy its code, so a wrong signature in an
article is a compiler error handed out with authority. This checks each block the way rustdoc
checks doc examples:

- ```rust                      must compile.
- ```rust,compile_fail,E0599   must FAIL to compile, with that error code (with several, as in
                               `compile_fail,E0499,E0503`, all of them). This is how an article
                               shows the error a model will actually see, and proves it.
- ```rust,ignore / ```text     are not checked (fragments, output, pseudo-code).
- ```rust,no_run               compiles but is not run with --run (a server that serves forever).

With `--run`, each compiling block is also run, the way rustdoc runs doc examples: its `main`,
then its `#[test]` functions if it has any, each with a timeout. Compiling proves the API; it does
not prove that an article's claims about behavior (a status code, an order, a message) are true,
and those are the claims a model acts on. Write runnable blocks with `assert!`s for exactly that.

Blocks are compiled as edition 2024, what `cargo new` produces, and in test mode
(`cargo test --no-run`): a plain build strips `#[test]` and `#[tokio::test]` functions before
type checking, so an article's example tests would otherwise never be checked at all. A block
without `fn main` is compiled as items, with an empty `main` added. Every block must carry its own
`use` lines: it is read on its own, and so is it compiled.

Dependencies come from `check.toml` in the article's directory (a `[dependencies]` table), so
the crate version the article is checked against is written down next to it.

Usage:
    python scripts/check_library.py                       # the library in haico's store
    python scripts/check_library.py path/to/library [--work C:/Temp/library-check]
    python scripts/check_library.py --only mio/basics     # one article
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FENCE = re.compile(r"^```(.*)$")


def default_library() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "Haico" / "library"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Haico" / "library"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "haico" / "library"


def blocks(text: str):
    """(line number, info string, code) for every fenced block."""
    out, info, start, lines = [], None, 0, []
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line.strip()) if line.lstrip().startswith("```") else None
        if info is None and match:
            info, start, lines = match.group(1).strip(), number, []
        elif info is not None and line.strip() == "```":
            out.append((start, info, "\n".join(lines) + "\n"))
            info = None
        elif info is not None:
            lines.append(line)
    return out


def runnable(info: str) -> bool:
    return "no_run" not in [p.strip() for p in info.split(",")]


def kind(info: str):
    """('pass'|'fail'|None, the error codes a compile_fail block must produce)."""
    parts = [p.strip() for p in info.split(",")]
    if not parts or parts[0] != "rust":
        return None, None
    if "ignore" in parts:
        return None, None
    if "compile_fail" in parts:
        return "fail", [p for p in parts if re.fullmatch(r"E\d{4}", p)]
    return "pass", None


def program(code: str) -> str:
    if re.search(r"^\s*(pub\s+)?(async\s+)?fn\s+main\s*\(", code, re.M):
        return "#![allow(unused, dead_code)]\n" + code
    return "#![allow(unused, dead_code)]\n" + code + "\nfn main() {}\n"


def cargo(work: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", "test", "--no-run", *args, "--quiet", "--color", "never"],
        cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def run_binary(path: Path, timeout: float) -> str | None:
    """None when it exits 0; otherwise what went wrong, with the end of its output."""
    try:
        result = subprocess.run([str(path)], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # What it printed before the timeout; bytes on some platforms whatever `text` says.
        tail = "".join(p.decode("utf-8", "replace") if isinstance(p, bytes) else p
                       for p in (e.stdout, e.stderr) if p)
        return f"timed out after {timeout:.0f} s (mark it no_run if it serves forever)\n{indent(tail)}"
    if result.returncode != 0:
        return f"exited with {result.returncode}\n{indent(result.stdout + result.stderr)}"
    return None


def built_executables(work: Path, *args) -> dict:
    """{example name: executable} for a build with --message-format=json."""
    result = subprocess.run(
        ["cargo", *args, "--quiet", "--message-format=json"],
        cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    found = {}
    for line in result.stdout.splitlines():
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            found[message["target"]["name"]] = Path(message["executable"])
    return found


def run_blocks(work: Path, rel: str, runs: list, timeout: float) -> list:
    """Run each (line, name, has_tests) block's main, then its tests."""
    problems = []
    mains = built_executables(work, "build", "--examples")
    tests = built_executables(work, "test", "--no-run", "--examples") if any(t for _, _, t in runs) else {}
    for line, name, has_tests in runs:
        if name in mains and (why := run_binary(mains[name], timeout)):
            problems.append(f"{rel}.md:{line}: main {why}")
        if has_tests and name in tests and (why := run_binary(tests[name], timeout)):
            problems.append(f"{rel}.md:{line}: tests {why}")
    return problems


TEST_ATTRIBUTE = re.compile(r"#\[(\w+::)?test(\(.*\))?\]")


def check_article(article: Path, library: Path, work_root: Path, run: bool = False, timeout: float = 60) -> list:
    rel = article.relative_to(library).with_suffix("").as_posix()
    manifest = article.parent / "check.toml"
    dependencies = manifest.read_text(encoding="utf-8") if manifest.exists() else "[dependencies]\n"

    all_blocks = blocks(article.read_text(encoding="utf-8"))
    checked = [(line, *kind(info), code) for line, info, code in all_blocks]
    checked = [c for c in checked if c[1] is not None]
    no_run = {line for line, info, _ in all_blocks if not runnable(info)}
    if not checked:
        return []

    # One project per article, reused between runs so the dependencies build once.
    work = work_root / rel.replace("/", "__")
    examples = work / "examples"
    if examples.exists():
        shutil.rmtree(examples)
    examples.mkdir(parents=True)
    (work / "src").mkdir(exist_ok=True)
    (work / "src" / "lib.rs").write_text("", encoding="utf-8")
    (work / "Cargo.toml").write_text(
        '[package]\nname = "library_check"\nversion = "0.0.0"\nedition = "2024"\npublish = false\n\n'
        + dependencies,
        encoding="utf-8",
    )

    problems = []
    passing = []
    for line, expect, code_expected, code in checked:
        name = f"line_{line}"
        (examples / f"{name}.rs").write_text(program(code), encoding="utf-8")
        if expect == "pass":
            passing.append((line, name))

    # The passing blocks together first: one build, and any failure names its example.
    if passing:
        result = cargo(work, "--examples")
        if result.returncode != 0:
            # Rebuild one by one to say exactly which blocks are broken.
            for line, name in passing:
                single = cargo(work, "--example", name)
                if single.returncode != 0:
                    problems.append(f"{rel}.md:{line}: does not compile\n{indent(single.stderr)}")
        if run and not problems:
            codes = {line: code for line, expect, _, code in checked if expect == "pass"}
            runs = [(line, name, bool(TEST_ATTRIBUTE.search(codes[line])))
                    for line, name in passing if line not in no_run]
            problems += run_blocks(work, rel, runs, timeout)

    for line, expect, code_expected, _ in checked:
        if expect != "fail":
            continue
        result = cargo(work, "--example", f"line_{line}")
        if result.returncode == 0:
            problems.append(f"{rel}.md:{line}: marked compile_fail but compiles")
        elif code_expected and not set(code_expected) <= headline_codes(result.stderr):
            found = sorted(headline_codes(result.stderr)) or ["no error code"]
            problems.append(
                f"{rel}.md:{line}: expected {', '.join(code_expected)}, got {', '.join(found)}\n{indent(result.stderr)}"
            )

    print(f"{rel}: {len(checked)} block(s), {len(problems)} problem(s)")
    return problems


def headline_codes(stderr: str) -> set:
    """The codes of the errors rustc reported: only lines that *start* with `error[E....]`.

    Searching the whole output is wrong: rustc echoes the offending source line, and an example
    that quotes its expected error in a comment would then "find" that code in its own quotation.
    That happened: a block claiming E0594 passed while rustc actually reported E0596.
    """
    return set(re.findall(r"(?m)^error\[(E\d{4})\]", stderr))


def indent(text: str, limit: int = 40) -> str:
    lines = text.strip().splitlines()
    shown = lines[:limit] + (["..."] if len(lines) > limit else [])
    return "\n".join("    " + l for l in shown)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", nargs="?", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "library-check")
    parser.add_argument("--only", help="check one article, by id (mio/basics), or a directory (mio)")
    parser.add_argument("--run", action="store_true", help="also run every block not marked no_run")
    parser.add_argument("--timeout", type=float, default=60, help="seconds per run (default 60)")
    args = parser.parse_args()

    library = (args.library or default_library()).resolve()
    articles = sorted(library.rglob("*.md"))
    if args.only:
        ids = {a: a.relative_to(library).with_suffix("").as_posix() for a in articles}
        articles = [a for a in articles if ids[a] == args.only or ids[a].startswith(args.only.rstrip("/") + "/")]
    if not articles:
        print(f"no articles found under {library}", file=sys.stderr)
        return 1

    problems = []
    for article in articles:
        problems += check_article(article, library, args.work, args.run, args.timeout)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
