#!/usr/bin/env python3
"""Turn haico --trace files into a chat fine-tuning dataset.

A trace is raw material: one line per model call, each carrying the whole request as
sent. This picks the runs worth learning from and turns them into training examples in
the shape TRL, axolotl, OpenAI fine-tuning and most others read:

    {"messages": [...], "tools": [...]}

How a run becomes examples
--------------------------
Every main-conversation call repeats the history before it, so training on each one
would repeat the early turns many times over. Calls are therefore chained: a call whose
request is the previous call's request plus its reply (plus whatever tool results and
user messages followed) continues the same stretch, and only the last call of each
stretch becomes an example — with its reply appended. A request that does *not* extend
the one before starts a new stretch: that is a compaction, which replaced the history
with a summary, or a different conversation altogether (every reviewer call is its own).

The history inside that last request is what haico sent, and under reasoning =
"compress" or "drop" its earlier assistant turns carry rewritten or missing thinking.
By default (--history genuine) each earlier assistant turn is swapped back for the reply
exactly as that call recorded it, so the model is trained on what it actually wrote.

Picking runs
------------
Runs are filtered on their main conversation: its exit code (0 by default), how many
calls it took (--max-turns), the largest single reply (--max-turn-tokens), the total it
generated (--max-run-tokens) and its wall time (--max-seconds). --top K then keeps the K
tightest of what is left, ranked by turns, then tokens, then time. Token counts come
from the server's usage report when there is one, else an estimate at four characters
per token.

Examples are dropped when a reply in them was cut off by max_tokens (finish_reason
"length"), when a tool call's arguments are not valid JSON (haico repairs those before
running them; the model should not learn to write them), and when they carry images,
unless --keep-images.

Questions and answers
---------------------
--qa exports each library lookup as two messages: the question the librarian was asked
and its final answer, with nothing between — no system prompt, tools, articles or
reasoning. That is the shape for teaching a model what the library says, so it stops
needing to ask. Answers that only make sense with the articles at hand are dropped: "the
library does not cover this", and anything citing its source ("the article", a section
id). So are exact repeats of a question and answer seen earlier. haico --trace-lib writes
the same examples as it goes, with the same filters — keep the patterns in step with
qa_rejection in src/trace.rs.

Usage
-----
    python scripts/trace_to_dataset.py runs.jsonl -o dataset.jsonl
    python scripts/trace_to_dataset.py nightly/*.jsonl --max-turns 12 --max-turn-tokens 4000 --top 50
    python scripts/trace_to_dataset.py runs.jsonl --caller reviewer -o reviewer.jsonl
    python scripts/trace_to_dataset.py runs.jsonl --anonymize --reasoning drop --meta
    python scripts/trace_to_dataset.py traces/*.jsonl --qa --any-exit -o library-qa.jsonl

A summary of what was kept and why the rest was not goes to stderr.
"""

import argparse
import copy
import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime

CALLERS = ["main", "research", "library", "reviewer", "compressor", "compactor", "summarizer"]


# --------------------------------------------------------------------------- reading

def read_traces(paths):
    """Every line of every file, grouped by run id, in the order runs first appear."""
    runs = OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves half a line; the rest of the file is fine.
                    print(f"{path}:{number}: skipped an unreadable line", file=sys.stderr)
                    continue
                run = runs.setdefault(entry.get("run", "?"), {"start": None, "end": None, "calls": []})
                kind = entry.get("type")
                if kind == "start":
                    run["start"] = entry
                elif kind == "end":
                    run["end"] = entry
                elif kind == "call":
                    run["calls"].append(entry)
    return runs


# --------------------------------------------------------------------------- measuring

def reply_tokens(call):
    """Tokens the model generated for this call: the server's count, else an estimate."""
    usage = call.get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    message = call.get("response") or {}
    text = (message.get("content") or "") + (message.get("reasoning_content") or message.get("reasoning") or "")
    for tool_call in message.get("tool_calls") or []:
        text += (tool_call.get("function") or {}).get("arguments") or ""
    return len(text) // 4


def run_stats(run):
    """How long and how wordy the run's main conversation was."""
    main = [c for c in run["calls"] if c.get("caller") == "main" and "response" in c]
    tokens = [reply_tokens(c) for c in main]
    seconds = None
    try:
        start = datetime.fromisoformat(run["start"]["time"])
        end = datetime.fromisoformat(run["end"]["time"])
        seconds = (end - start).total_seconds()
    except (TypeError, KeyError, ValueError):
        # No start or end line (a run killed outright): fall back to time spent in calls.
        seconds = sum(c.get("elapsed_ms", 0) for c in run["calls"]) / 1000
    return {
        "turns": len(main),
        "max_turn_tokens": max(tokens, default=0),
        "run_tokens": sum(tokens),
        "seconds": seconds,
        "exit_code": (run["end"] or {}).get("exit_code"),
    }


def select_runs(runs, args, dropped):
    kept = []
    for run_id, run in runs.items():
        stats = run_stats(run)
        run["stats"] = stats
        if not args.any_exit:
            if run["end"] is None:
                dropped["run: no end line (did not exit cleanly)"] += 1
                continue
            if stats["exit_code"] not in args.exit_code:
                dropped[f"run: exit code {stats['exit_code']}"] += 1
                continue
        if stats["turns"] == 0:
            dropped["run: no main-conversation replies"] += 1
            continue
        if args.max_turns is not None and stats["turns"] > args.max_turns:
            dropped["run: over --max-turns"] += 1
            continue
        if args.max_turn_tokens is not None and stats["max_turn_tokens"] > args.max_turn_tokens:
            dropped["run: over --max-turn-tokens"] += 1
            continue
        if args.max_run_tokens is not None and stats["run_tokens"] > args.max_run_tokens:
            dropped["run: over --max-run-tokens"] += 1
            continue
        if args.max_seconds is not None and stats["seconds"] > args.max_seconds:
            dropped["run: over --max-seconds"] += 1
            continue
        kept.append((run_id, run))

    if args.top is not None and len(kept) > args.top:
        kept.sort(key=lambda item: (item[1]["stats"]["turns"], item[1]["stats"]["run_tokens"], item[1]["stats"]["seconds"]))
        dropped["run: outside --top"] += len(kept) - args.top
        kept = kept[: args.top]
    return kept


# --------------------------------------------------------------------------- chaining

def non_system(messages):
    return [m for m in messages if m.get("role") != "system"]


def key(message):
    """What identifies a message across requests. An assistant turn is known by what it
    said and which calls it made, never by its reasoning — that is the part a reasoning
    policy rewrites between the reply and the next request."""
    role = message.get("role")
    if role == "assistant":
        ids = tuple(tc.get("id") for tc in message.get("tool_calls") or [])
        return (role, message.get("content") or "", ids)
    content = message.get("content")
    return (role, content if isinstance(content, str) else json.dumps(content, sort_keys=True))


def extends(previous, call):
    """Whether `call`'s request is `previous`'s request plus its reply and what followed."""
    before = [key(m) for m in non_system(previous["request"].get("messages", []))]
    after = [key(m) for m in non_system(call["request"].get("messages", []))]
    expected = before + [key(previous["response"])]
    return after[: len(expected)] == expected


def stretches(calls):
    """Consecutive calls of one conversation, split wherever the history was replaced."""
    groups = []
    for call in calls:
        if "response" not in call:
            continue  # failed calls produced nothing and are not in any later history
        if groups and extends(groups[-1][-1], call):
            groups[-1].append(call)
        else:
            groups.append([call])
    return groups


# --------------------------------------------------------------------------- building

def build_example(stretch, args):
    """The last call of the stretch as one training example, or (None, reason)."""
    last = stretch[-1]
    messages = copy.deepcopy(last["request"].get("messages", []))

    if args.history == "genuine":
        # Call k's reply sits in the last request right after call k's own history, so
        # its position there is the length of call k's non-system messages.
        system_count = len(messages) - len(non_system(messages))
        for call in stretch[:-1]:
            position = system_count + len(non_system(call["request"].get("messages", [])))
            if position < len(messages) and messages[position].get("role") == "assistant":
                messages[position] = copy.deepcopy(call["response"])

    final = copy.deepcopy(last["response"])
    final.setdefault("role", "assistant")
    messages.append(final)

    # Every reply in the example, with how it ended, for the checks below. Under
    # --history as-sent the earlier ones are what was sent, but their endings still stand.
    for call in stretch:
        if call.get("finish_reason") == "length":
            return None, "example: a reply was cut off (finish_reason length)"

    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                arguments = (tool_call.get("function") or {}).get("arguments")
                try:
                    if not isinstance(json.loads(arguments), dict):
                        raise ValueError
                except (TypeError, ValueError):
                    return None, "example: tool call with invalid arguments"
        if not args.keep_images and isinstance(message.get("content"), list):
            if any(part.get("type") == "image_url" for part in message["content"] if isinstance(part, dict)):
                return None, "example: carries an image"

    for message in messages:
        # Some servers spell it `reasoning`; trainers and chat templates read `reasoning_content`.
        if "reasoning" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("reasoning")
        if args.reasoning == "drop":
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
        if message.get("reasoning_content") == "":
            message.pop("reasoning_content")

    example = {"messages": messages}
    # The tools the model was shown. A stretch can end on a call with none — research's
    # last answer is asked for without tools — while earlier turns in it did call them.
    tools = next((c["request"]["tools"] for c in reversed(stretch) if c["request"].get("tools")), None)
    if tools:
        example["tools"] = tools
    return example, None


# --------------------------------------------------------------------------- question and answer

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)

# The librarian telling the reader the articles had nothing. Right with the library behind
# it, and exactly wrong as a thing to learn to say from memory.
NOT_COVERED = re.compile(
    r"\b(?:library|articles?)\b[^.\n]{0,40}\b(?:does not|doesn't|do not|don't)\b[^.\n]{0,20}"
    r"\b(?:cover|mention|say|describe|answer|include)", re.I)

# An answer pointing at what it was read from: "the article shows", "as I read", a section
# id like tokio/select#2. Trained without the lookup, that teaches citing a source the model
# never had. "the library" is included although a crate is a library too — the librarian
# says it of its own shelves far more often than of a crate.
REFERS_TO_SOURCE = re.compile(
    r"\bthe articles?\b|\bthis article\b|\bthe library\b|\b(?:I|we) (?:have )?read\b"
    r"|\bsections? (?:I|you) read\b|\b[\w.-]+/[\w.-]+#\d+\b", re.I)


def build_qa(stretch):
    """The stretch's question and final answer as one two-message example, or (None, reason).

    For putting what the library knows into the weights: the question as the user asked it,
    the answer as if known outright — no system prompt, no tools, no articles, no reasoning
    (which was done with the articles in view and talks about them)."""
    first = stretch[0]["request"].get("messages", [])
    question = next((m.get("content") for m in first if m.get("role") == "user"), None)
    if not isinstance(question, str) or not question.strip():
        return None, "qa: no question"

    last = stretch[-1]
    if last.get("finish_reason") == "length":
        return None, "qa: the answer was cut off (finish_reason length)"
    if last["response"].get("tool_calls"):
        return None, "qa: ended on a lookup, not an answer"
    answer = THINK_BLOCK.sub("", last["response"].get("content") or "").strip()
    if not answer:
        return None, "qa: empty answer"
    if NOT_COVERED.search(answer):
        return None, "qa: the library did not cover it"
    if REFERS_TO_SOURCE.search(answer):
        return None, "qa: the answer refers to the articles"

    return {"messages": [{"role": "user", "content": question.strip()},
                         {"role": "assistant", "content": answer}]}, None


# --------------------------------------------------------------------------- anonymizing

USER_DIRS = [
    (re.compile(r"([A-Za-z]:\\+Users\\+)[^\\/\"'\s]+"), r"\1user"),
    (re.compile(r"(/home/)[^/\"'\s]+"), r"\1user"),
    (re.compile(r"(/Users/)[^/\"'\s]+"), r"\1user"),
]
SCRATCH = re.compile(r"[^\s\"'`]*?haico-[0-9a-f]{8}")


def anonymize(value, cwd, project_path):
    """Replace machine-specific paths in every string of `value`. Best effort: it knows
    the project directory, haico's scratch directories and home directories, and nothing
    else — read the output before it leaves the machine."""
    if isinstance(value, str):
        if cwd:
            for spelling in {cwd, cwd.replace("\\", "/"), cwd.replace("\\", "\\\\")}:
                value = value.replace(spelling, project_path)
        value = SCRATCH.sub("/tmp/haico-scratch", value)
        for pattern, replacement in USER_DIRS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [anonymize(v, cwd, project_path) for v in value]
    if isinstance(value, dict):
        return {k: anonymize(v, cwd, project_path) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Turn haico --trace files into a chat fine-tuning dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n", 1)[1],
    )
    parser.add_argument("traces", nargs="+", help="trace files written by haico --trace")
    parser.add_argument("-o", "--output", help="dataset file to write (default: stdout)")
    parser.add_argument("--caller", choices=CALLERS,
                        help="which conversation to export (default: main, or library with --qa)")
    parser.add_argument("--qa", action="store_true",
                        help="export each library lookup as just its question and final answer")

    runs = parser.add_argument_group("choosing runs")
    runs.add_argument("--exit-code", type=int, action="append",
                      help="keep runs that exited with this code; repeatable (default: 0)")
    runs.add_argument("--any-exit", action="store_true",
                      help="keep runs whatever they exited with, including REPL sessions and killed runs")
    runs.add_argument("--max-turns", type=int, help="at most this many main-conversation calls")
    runs.add_argument("--max-turn-tokens", type=int, help="no single reply longer than this many tokens")
    runs.add_argument("--max-run-tokens", type=int, help="at most this many tokens generated in total")
    runs.add_argument("--max-seconds", type=float, help="at most this long from start to end")
    runs.add_argument("--top", type=int, help="then keep only the K tightest runs (fewest turns, then tokens, then time)")

    shape = parser.add_argument_group("shaping examples")
    shape.add_argument("--history", choices=["genuine", "as-sent"], default="genuine",
                       help="earlier assistant turns as each call recorded them (default), or as haico sent them")
    shape.add_argument("--reasoning", choices=["keep", "drop"], default="keep",
                       help="keep the chain of thought in assistant turns (default) or strip it")
    shape.add_argument("--keep-images", action="store_true", help="keep examples that carry images")
    shape.add_argument("--anonymize", action="store_true",
                       help="replace the project path, haico's scratch paths and home directories")
    shape.add_argument("--project-path", default="/workspace/project",
                       help="what --anonymize puts in place of the project path (default: /workspace/project)")
    shape.add_argument("--meta", action="store_true",
                       help="add a 'meta' object (run, caller, exit code, turns, tokens) to each example")
    args = parser.parse_args(argv)
    if args.caller is None:
        args.caller = "library" if args.qa else "main"
    elif args.qa and args.caller != "library":
        parser.error("--qa works on the librarian's conversation only (--caller library)")
    if args.exit_code is None:
        args.exit_code = [0]

    dropped = Counter()
    all_runs = read_traces(args.traces)
    kept_runs = select_runs(all_runs, args, dropped)

    examples = []
    seen_qa = set()
    for run_id, run in kept_runs:
        calls = [c for c in run["calls"] if c.get("caller") == args.caller]
        for stretch in stretches(calls):
            example, reason = build_qa(stretch) if args.qa else build_example(stretch, args)
            if example is None:
                dropped[reason] += 1
                continue
            if args.qa:
                # A model stuck on one error asks the same thing again and gets the same answer.
                pair = tuple(m["content"] for m in example["messages"])
                if pair in seen_qa:
                    dropped["qa: same question and answer as an earlier one"] += 1
                    continue
                seen_qa.add(pair)
            if args.anonymize:
                cwd = (run["start"] or {}).get("cwd")
                example = anonymize(example, cwd, args.project_path)
            if args.meta:
                example["meta"] = {"run": run_id, "caller": args.caller, **run["stats"]}
            examples.append(example)

    out = open(args.output, "w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    try:
        for example in examples:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out.close()

    print(f"runs: {len(all_runs)} read, {len(kept_runs)} kept", file=sys.stderr)
    for run_id, run in kept_runs:
        s = run["stats"]
        print(f"  {run_id}  turns {s['turns']}  max turn {s['max_turn_tokens']} tok  "
              f"total {s['run_tokens']} tok  {s['seconds']:.0f}s", file=sys.stderr)
    print(f"examples: {len(examples)} written ({args.caller})", file=sys.stderr)
    for reason, count in sorted(dropped.items()):
        print(f"  dropped {count}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
