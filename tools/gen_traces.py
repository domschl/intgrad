#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""gen_traces.py — tool-use trajectory generation via the local teacher
(Decision D3: llama.cpp server at 127.0.0.1:8180, OpenAI-compatible API).

Runs a real agent loop: teacher decides call-vs-answer, mock tools execute
locally and deterministically, results are fed back. Only trajectories that
terminate with a valid answer AND whose tool calls match the JSON schema
survive (rejection sampling, plan §4.3).

Output: JSONL, one trajectory per line:
  {"id": int, "messages": [{"role": "system|user|assistant|tool",
     "content": str, "call": {"name":..., "args":...}|None,
     "tool_ok": bool|None}, ...]}

Convert to the byte-stream wire format (protocol.h / ig-pretty tags) with:
  --wire out.bin

No dependencies beyond stdlib.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request

URL = "http://127.0.0.1:8180/v1/chat/completions"

TOOLS = {
    "get_weather": {
        "desc": "get_weather(city: str) -> current weather for a city",
    },
    "calc": {
        "desc": "calc(expression: str) -> value of an arithmetic expression (+ - * / ( ) numbers)",
    },
    "lookup_user": {
        "desc": "lookup_user(name: str) -> user record {id, email} or error",
    },
}

USERS = {
    "alice": {"id": 1001, "email": "alice@example.org"},
    "bob": {"id": 1002, "email": "bob@example.org"},
    "carla": {"id": 1003, "email": "carla@example.net"},
}

SYS_PROMPT = (
    "You are an assistant that can use these tools:\n"
    + "\n".join("- " + t["desc"] for t in TOOLS.values())
    + "\n\nTo call a tool, reply with ONLY a JSON object:"
    '\n{"type":"call","name":"<tool>","args":{...}}'
    "\nYou will then receive {\"type\":\"result\",...}. You may call tools"
    " again. When you can answer, reply with ONLY:"
    '\n{"type":"answer","text":"..."}'
    "\nIf no tool is needed, answer directly. Never invent results."
)

TASKS = [
    "What's the weather in {city}?",
    "What is {expr}?",
    "Do you have a user named {user}? What's their email?",
    "Is it warmer in {city} or {city2}? Answer using the weather tool for both.",
    "Look up {user} and also check the weather in {city}; summarize both in one sentence.",
    "What is 17*{n} + 3?",
    "Thanks, no task — just say hi.",
    "Who won the 2030 world cup? (no tool for that — answer honestly)",
]
CITIES = ["Paris", "Berlin", "Madrid", "Lisbon", "Vienna", "Rome", "Amsterdam", "Prague"]


def mock_exec(name, args):
    """Deterministic sandboxed tools. Returns (ok, payload)."""
    if name == "get_weather":
        city = str(args.get("city", ""))
        h = int(hashlib.sha256(city.lower().encode()).hexdigest(), 16)
        conds = ["sunny", "cloudy", "rain", "clear", "windy"]
        return True, {"temp_c": 5 + h % 28, "condition": conds[h % 5]}
    if name == "calc":
        expr = str(args.get("expression", ""))
        if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
            return False, {"error": "invalid characters"}
        try:
            return True, {"value": eval(expr, {"__builtins__": {}}, {})}
        except Exception as e:  # noqa: BLE001
            return False, {"error": str(e)}
    if name == "lookup_user":
        u = USERS.get(str(args.get("name", "")).lower())
        return (True, u) if u else (False, {"error": "not found"})
    return False, {"error": "unknown tool"}


def teacher(model, messages, temperature):
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def parse_step(text):
    """Extract the first JSON object from teacher output."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("type") == "call" and obj.get("name") in TOOLS:
        return obj
    if obj.get("type") == "answer" and isinstance(obj.get("text"), str):
        return obj
    return None


def gen_one(teacher_model, rng, temperature):
    task = rng.choice(TASKS).format(
        city=rng.choice(CITIES), city2=rng.choice(CITIES),
        expr=f"{rng.randint(2, 40)}*{rng.randint(2, 20)}-{rng.randint(1, 9)}",
        user=rng.choice(list(USERS)) if rng.random() < 0.7
        else "nobody",
        n=rng.randint(2, 9),
    )
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": task}]
    trace = [{"role": "system", "content": SYS_PROMPT},
             {"role": "user", "content": task}]
    for _ in range(4):
        try:
            out = teacher(teacher_model, messages, temperature)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  teacher error: {e}\n")
            return None
        step = parse_step(out)
        if step is None:
            return None  # malformed protocol -> reject
        if step["type"] == "answer":
            trace.append({"role": "assistant", "content": step["text"],
                          "call": None, "tool_ok": None})
            return trace
        ok, payload = mock_exec(step["name"], step.get("args", {}))
        trace.append({"role": "assistant", "content": "",
                      "call": {"name": step["name"],
                               "args": step.get("args", {})},
                      "tool_ok": ok})
        trace.append({"role": "tool", "content": payload,
                      "call": None, "tool_ok": ok})
        messages.append({"role": "assistant", "content": json.dumps(step)})
        messages.append({"role": "user", "content": json.dumps(
            {"type": "result", "ok": ok, **({"result": payload} if ok
                                             else {"error": payload})})})
    return None  # no answer within budget -> reject


WIRE = {"system": "\x00\x1e", "user": "\x00\x1f", "eot": "\x00\x1d",
        "call": "\x00\x0b", "args": "\x00\x0c", "res": "\x00\x0d",
        "err": "\x00\x1c"}


def to_wire(trace):
    parts = []
    for m in trace:
        if m["role"] in ("system", "user"):
            parts.append(WIRE[m["role"]] + m["content"] + WIRE["eot"])
        elif m["role"] == "assistant":
            if m.get("call"):
                parts.append(WIRE["call"] + m["call"]["name"] + WIRE["args"]
                             + json.dumps(m["call"]["args"],
                                          separators=(",", ":"))
                             + WIRE["eot"])
            else:
                parts.append(m["content"] + WIRE["eot"])
        elif m["role"] == "tool":
            parts.append((WIRE["res"] if m["tool_ok"] else WIRE["err"])
                         + json.dumps(m["content"]) + WIRE["eot"])
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="trajectories to keep")
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--model", default="Qwen3.8-Next")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/traces.jsonl")
    ap.add_argument("--wire", default=None, help="also write byte-stream file")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept, attempted = [], 0
    while len(kept) < args.n and attempted < args.attempts:
        attempted += 1
        tr = gen_one(args.model, rng, args.temperature)
        if tr is None:
            sys.stderr.write(f"[{attempted}] reject\n")
            continue
        tid = len(kept)
        # materialize tool messages into the record
        record = []
        for m in tr:
            record.append(m)
        kept.append({"id": tid, "messages": record})
        sys.stderr.write(f"[{attempted}] keep ({len(record)} msgs)\n")

    with open(args.out, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{attempted} -> {args.out}")

    if args.wire and kept:
        # NOTE: wire conversion here is text-mode demo; the M4 dataset
        # builder (C) writes u8 shards + loss masks + provenance directly.
        blob = b"".join(to_wire(t["messages"]).encode("utf-8") for t in kept)
        with open(args.wire, "wb") as f:
            f.write(blob)
        print(f"wire stream -> {args.wire} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
