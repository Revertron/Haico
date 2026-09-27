"""Compile every Rust code block in haico's library articles.

An article is read by a model that is about to copy its code, so a wrong signature in an
article is a compiler error handed out with authority. This checks each block the way rustdoc
checks doc examples:

- ```rust                      must compile.
- ```rust,compile_fail,E0599   must FAIL to compile, with that error code. This is how an
                               article shows the error a model will actually see, and proves it.
- ```rust,ignore / ```text     are not checked (fragments, output, pseudo-code).

A block without `fn main` is compiled as items, with an empty `main` added. Every block must
carry its own `use` lines: it is read on its own, and so is it compiled.

Dependencies come from `check.toml` in the article's directory (a `[dependencies]` table), so
the crate version the article is checked against is written down next to it.

Usage:
    python scripts/check_library.py                       # the library in haico's store
    python scripts/check_library.py path/to/library [--work C:/Temp/library-check]
    python scripts/check_library.py --only mio/basics     # one article
"""

import argparse
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


def kind(info: str):
    """('pass'|'fail'|None, expected error code or None)."""
    parts = [p.strip() for p in info.split(",")]
    if not parts or parts[0] != "rust":
        return None, None
    if "ignore" in parts:
        return None, None
    if "compile_fail" in parts:
        code = next((p for p in parts if re.fullmatch(r"E\d{4}", p)), None)
        return "fail", code
    return "pass", None


def program(code: str) -> str:
    if re.search(r"^\s*(pub\s+)?(async\s+)?fn\s+main\s*\(", code, re.M):
        return "#![allow(unused, dead_code)]\n" + code
    return "#![allow(unused, dead_code)]\n" + code + "\nfn main() {}\n"


def cargo(work: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", *args, "--quiet", "--color", "never"],
        cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def check_article(article: Path, library: Path, work_root: Path) -> list:
    rel = article.relative_to(library).with_suffix("").as_posix()
    manifest = article.parent / "check.toml"
    dependencies = manifest.read_text(encoding="utf-8") if manifest.exists() else "[dependencies]\n"

    checked = [(line, *kind(info), code) for line, info, code in blocks(article.read_text(encoding="utf-8"))]
    checked = [c for c in checked if c[1] is not None]
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
        '[package]\nname = "library_check"\nversion = "0.0.0"\nedition = "2021"\npublish = false\n\n'
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
        result = cargo(work, "build", "--examples")
        if result.returncode != 0:
            # Rebuild one by one to say exactly which blocks are broken.
            for line, name in passing:
                single = cargo(work, "build", "--example", name)
                if single.returncode != 0:
                    problems.append(f"{rel}.md:{line}: does not compile\n{indent(single.stderr)}")

    for line, expect, code_expected, _ in checked:
        if expect != "fail":
            continue
        result = cargo(work, "build", "--example", f"line_{line}")
        if result.returncode == 0:
            problems.append(f"{rel}.md:{line}: marked compile_fail but compiles")
        elif code_expected and f"error[{code_expected}]" not in result.stderr:
            found = sorted(set(re.findall(r"error\[E\d{4}\]", result.stderr))) or ["no error code"]
            problems.append(
                f"{rel}.md:{line}: expected {code_expected}, got {', '.join(found)}\n{indent(result.stderr)}"
            )

    print(f"{rel}: {len(checked)} block(s), {len(problems)} problem(s)")
    return problems


def indent(text: str, limit: int = 40) -> str:
    lines = text.strip().splitlines()
    shown = lines[:limit] + (["..."] if len(lines) > limit else [])
    return "\n".join("    " + l for l in shown)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", nargs="?", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "library-check")
    parser.add_argument("--only", help="check one article, by id (mio/basics)")
    args = parser.parse_args()

    library = (args.library or default_library()).resolve()
    articles = sorted(library.rglob("*.md"))
    if args.only:
        articles = [a for a in articles if a.relative_to(library).with_suffix("").as_posix() == args.only]
    if not articles:
        print(f"no articles found under {library}", file=sys.stderr)
        return 1

    problems = []
    for article in articles:
        problems += check_article(article, library, args.work)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
