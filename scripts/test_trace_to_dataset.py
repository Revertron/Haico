"""Tests for trace_to_dataset.py. Run with: python -m unittest discover scripts"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr

import trace_to_dataset as ttd

SYSTEM = {"role": "system", "content": "You are haico."}
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]


def call(caller, messages, response, finish="stop", tokens=10, tools=TOOLS):
    return {"type": "call", "caller": caller, "model": "m",
            "request": {"messages": messages, "tools": tools if caller == "main" else None},
            "response": response, "finish_reason": finish,
            "usage": {"completion_tokens": tokens}, "elapsed_ms": 1000}


def tool_call(id, arguments='{"file_path": "a"}'):
    return {"id": id, "type": "function", "function": {"name": "read", "arguments": arguments}}


def run(run_id, calls, exit_code=0):
    lines = [{"type": "start", "run": run_id, "time": "2026-09-24T10:00:00+02:00", "cwd": "C:\\proj", "task": "t"}]
    lines += [dict(c, run=run_id) for c in calls]
    if exit_code is not None:
        lines.append({"type": "end", "run": run_id, "time": "2026-09-24T10:01:00+02:00", "exit_code": exit_code})
    return lines


def one_task(run_id, turns=2, tokens=10, exit_code=0, finish="stop", arguments='{"file_path": "a"}'):
    """A main conversation of `turns` calls: each but the last reads a file."""
    history = [SYSTEM, {"role": "user", "content": "go"}]
    calls = []
    for i in range(turns - 1):
        reply = {"role": "assistant", "content": "", "reasoning_content": f"think {i}",
                 "tool_calls": [tool_call(f"c{i}", arguments)]}
        calls.append(call("main", list(history), reply, tokens=tokens, finish=finish))
        # What haico sends back: the reply with its reasoning compressed away.
        sent = dict(reply, reasoning_content="(compressed)")
        history += [sent, {"role": "tool", "tool_call_id": f"c{i}", "content": "file"}]
    calls.append(call("main", list(history), {"role": "assistant", "content": "done"}, tokens=tokens))
    return run(run_id, calls, exit_code)


def convert(lines, *args):
    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "trace.jsonl")
        out = os.path.join(dir, "out.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(l) + "\n" for l in lines)
        with redirect_stderr(io.StringIO()):
            ttd.main([path, "-o", out, *args])
        with open(out, encoding="utf-8") as f:
            return [json.loads(l) for l in f]


class Conversion(unittest.TestCase):
    def test_a_run_becomes_one_example_with_the_genuine_reasoning(self):
        [example] = convert(one_task("r", turns=3))
        roles = [m["role"] for m in example["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual(example["messages"][2]["reasoning_content"], "think 0")
        self.assertEqual(example["messages"][4]["reasoning_content"], "think 1")
        self.assertEqual(example["messages"][-1]["content"], "done")
        self.assertEqual(example["tools"], TOOLS)

    def test_as_sent_keeps_the_history_haico_sent(self):
        [example] = convert(one_task("r", turns=2), "--history", "as-sent")
        self.assertEqual(example["messages"][2]["reasoning_content"], "(compressed)")

    def test_a_compaction_splits_the_run_into_two_examples(self):
        lines = one_task("r", turns=2)
        calls = [l for l in lines if l["type"] == "call"]
        summary = [SYSTEM, {"role": "user", "content": "notes: read a"},
                   {"role": "assistant", "content": "ok"}, {"role": "user", "content": "continue"}]
        after = call("main", summary, {"role": "assistant", "content": "finished"})
        examples = convert(run("r", calls + [after]))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[1]["messages"][-1]["content"], "finished")

    def test_every_reviewer_call_is_its_own_example(self):
        lines = one_task("r", turns=2)
        verdict = {"role": "assistant", "content": '{"allow": true}'}
        lines[2:2] = [dict(call("reviewer", [SYSTEM, {"role": "user", "content": f"judge {i}"}], verdict), run="r")
                      for i in range(2)]
        examples = convert(lines, "--caller", "reviewer")
        self.assertEqual(len(examples), 2)
        self.assertNotIn("tools", examples[0])

    def test_broken_replies_are_dropped(self):
        self.assertEqual(convert(one_task("r", finish="length")), [])
        self.assertEqual(convert(one_task("r", arguments="{not json")), [])

    def test_runs_are_chosen_by_exit_and_size(self):
        lines = (one_task("ok", turns=2) + one_task("failed", exit_code=1)
                 + one_task("killed", exit_code=None) + one_task("long", turns=6)
                 + one_task("wordy", turns=2, tokens=5000))
        kept = lambda *a: [e["meta"]["run"] for e in convert(lines, "--meta", *a)]
        self.assertEqual(kept(), ["ok", "long", "wordy"])
        self.assertEqual(kept("--max-turns", "3"), ["ok", "wordy"])
        self.assertEqual(kept("--max-turn-tokens", "100"), ["ok", "long"])
        self.assertEqual(kept("--top", "1"), ["ok"])
        self.assertEqual(sorted(kept("--any-exit")), ["failed", "killed", "long", "ok", "wordy"])

    def test_qa_keeps_the_question_and_final_answer_only(self):
        lines = one_task("r", turns=2)
        librarian = [SYSTEM, {"role": "user", "content": "mio: register a TcpListener?"}]
        lookup = {"role": "assistant", "content": "", "reasoning_content": "search it",
                  "tool_calls": [tool_call("l0", '{"query": "register"}')]}
        answer = {"role": "assistant", "content": "<think>x</think>Call `registry.register`.",
                  "reasoning_content": "from mio/net#1"}
        history = librarian + [lookup, {"role": "tool", "tool_call_id": "l0", "content": "section"}]
        lines[2:2] = [dict(call("library", librarian, lookup), run="r"),
                      dict(call("library", history, answer), run="r")]
        [example] = convert(lines, "--qa")
        self.assertEqual(example, {"messages": [
            {"role": "user", "content": "mio: register a TcpListener?"},
            {"role": "assistant", "content": "Call `registry.register`."}]})

    def test_qa_drops_answers_that_lean_on_the_articles(self):
        answers = ["The library does not cover mio's Registry.",
                   "The article shows `registry.register`.",
                   "See mio/net#1: call `register`.",
                   "Call `register`.", "Call `register`."]
        lines = one_task("r", turns=2)
        lines[2:2] = [dict(call("library", [SYSTEM, {"role": "user", "content": "q"}],
                                {"role": "assistant", "content": a}), run="r") for a in answers]
        examples = convert(lines, "--qa")
        self.assertEqual([e["messages"][1]["content"] for e in examples], ["Call `register`."])

    def test_anonymizing_replaces_machine_paths(self):
        value = {"a": "in C:\\proj\\src and C:/proj, scratch G:\\Temp\\haico-0a1b2c3d, C:\\Users\\alice\\.cargo"}
        clean = ttd.anonymize(value, "C:\\proj", "/workspace/project")["a"]
        self.assertEqual(clean, "in /workspace/project\\src and /workspace/project, "
                                "scratch /tmp/haico-scratch, C:\\Users\\user\\.cargo")


if __name__ == "__main__":
    unittest.main()
