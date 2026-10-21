#!/usr/bin/env python3
"""PreToolUse guard for the MoE FP8 decode (e05c6c03) /goal arms (1020). Usage: guard.py <arm name>.

Refuses a tool call that
  - reads, fetches or copies the reference implementation (the KDA-0.5 kernel behind kda05-ref, the
    KDA / contest-winner repositories) -- kda05-ref may be run, not read;
  - touches the acceptance tool's files on the node (moe-accept may be run, not read or changed);
  - reads the FlashInfer / TensorRT-LLM fused-MoE sources (those libraries may only be run);
  - touches any path in this machine's home directory outside the allowed tool directories, any other
    /Users/Shared directory than this arm's own, or other sessions' scratch directories.
Write/Edit content is not scanned (comments may mention the reference); only their target path is.
Exit code 2 with a reason on stderr is how Claude Code is told to refuse the call.
"""
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
ARM = sys.argv[1]
ARM_DIR = f"/Users/Shared/moe-goal-1020-{ARM}"
OWN_SCRATCH = f"-Users-Shared-moe-goal-1020-{ARM}-ws"

FORBIDDEN_TEXT = [
    r"\.kda05ref", r"kda05_ref", r"mlsys2026-flashinfer-contest", r"humanfia/", r"kda-for-kda", r"NVlabs/kda",
    r"github\.com/[^\s'\"]*kda", r"huggingface\.co/[^\s'\"]*kda", r"kachua-mlsys", r"flashinfer-challenge",
    r"/opt/moe-accept", r"moe-accept-cache", r"/usr/local/bin/moe-accept",
    r"flashinfer/fused_moe", r"flashinfer/data/csrc", r"trtllm[^\s'\"]*\.(cu|cuh|h|cpp|py)\b", r"getsource",
    r"inspect\.getfile", r"__file__\)?[^\n]*fused_moe", r"github\.com/[^\s'\"]*(flashinfer|TensorRT-LLM|wombat)",
]
READ_TOOL = (r"\b(cat|less|more|head|tail|sed|awk|vi|vim|nano|strings|xxd|od|cp|mv|scp|rsync|tar|zip|grep|rg|"
             r"base64|file|which|type|python[0-9.]*\s+-c|bash\s+-x|ls)\b[^|;&\n]*\b(kda05-ref|moe-accept)\b")
ALLOWED_HOME = [
    os.path.join(HOME, ".tsh"),
    os.path.join(HOME, ".cybertronctl"),
    os.path.join(HOME, ".cache"),
    os.path.join(HOME, "Library", "Caches"),
    os.path.join(HOME, ".local", "bin"),
]
HOME_REF = re.compile(r"(?:" + re.escape(HOME) + r"|~(?=/)|\$HOME\b|\$\{HOME\})([^\s'\"`;|&)?,:!*\\]*)")
BARE_HOME = re.compile(r"\bcd(\s+(~|\$HOME|\$\{HOME\}))?\s*(?=$|[;&|'\"]|\\n)")
SCRATCH = re.compile(r"/tmp/claude-\d+/([^/\s'\"]*)")


def refuse(why):
    sys.stderr.write(f"Refused by the experiment's guard: {why}\n")
    sys.exit(2)


def main():
    try:
        call = json.load(sys.stdin)
    except Exception:
        return
    tool = call.get("tool_name", "")
    inp = dict(call.get("tool_input", {}) or {})
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        inp = {k: v for k, v in inp.items() if k in ("file_path", "notebook_path")}
    text = json.dumps(inp, ensure_ascii=False).replace("\\/", "/")
    for pat in FORBIDDEN_TEXT:
        if re.search(pat, text, re.IGNORECASE):
            refuse("this touches the reference implementation, contest submissions, or the acceptance tool's files "
                   "(kda05-ref and moe-accept may be run, not read or changed; library sources may not be read)")
    if re.search(READ_TOOL, text):
        refuse("kda05-ref and moe-accept may be run, but their files may not be read or changed")
    for m in SCRATCH.finditer(text):
        if m.group(1) and m.group(1) != OWN_SCRATCH:
            refuse("other sessions' scratch directories are off limits")
    if BARE_HOME.search(text):
        refuse(f"the home directory is outside your workspace; work only inside {ARM_DIR}/ws")
    for m in HOME_REF.finditer(text):
        path = os.path.normpath(HOME + m.group(1))
        if not any(path == p or path.startswith(p + os.sep) for p in ALLOWED_HOME):
            refuse(f"{path} is outside your workspace; work only inside {ARM_DIR}/ws")
    for m in re.finditer(r"/Users/Shared[^\s'\"`;|&)?,:!*\\]*", text):
        p = os.path.normpath(m.group(0))
        if not (p.startswith(ARM_DIR + "/ws") or p.startswith(ARM_DIR + "/cfg/skills")
                or p.startswith(ARM_DIR + "/cfg/projects") or p == ARM_DIR):
            refuse(f"{p} is outside your workspace; work only inside {ARM_DIR}/ws")


if __name__ == "__main__":
    main()
