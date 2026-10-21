#!/usr/bin/env python3
"""PreToolUse guard for the FA4 varlen bwd flame-chase arm (1020).

Refuses a tool call that
  - reads or fetches the reference implementation's source (flash-attn-4 / flash_attn.cute),
  - touches any path in this machine's home directory outside the arm's workspace and skills,
  - touches other sessions' scratch directories.
Write/Edit content is not scanned (comments may mention flash-attention); only their target path is.
Exit code 2 with a reason on stderr is how Claude Code is told to refuse the call.
"""
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
ARM_DIR = "/Users/Shared/fa4-bwd-fc-1020"
OWN_SCRATCH = "-Users-Shared-fa4-bwd-fc-1020-ws"

FORBIDDEN_TEXT = [
    r"site-packages/flash_attn", r"flash_attn/cute", r"flash_attn_4", r"flash-attn-4[^\s'\"]*\.(tar|whl|zip)",
    r"Dao-AILab", r"github\.com/[^\s'\"]*flash-?attention", r"inspect\.getsource", r"getsourcelines",
    r"pip\s+download[^|;&\n]*flash", r"unzip[^|;&\n]*flash",
]
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
            refuse("the reference implementation's source code is off limits (it may be imported and run, not read)")
    for m in SCRATCH.finditer(text):
        if m.group(1) and m.group(1) != OWN_SCRATCH:
            refuse("other sessions' scratch directories are off limits")
    if BARE_HOME.search(text):
        refuse(f"the home directory is outside your workspace; work only inside {ARM_DIR}/ws")
    for m in HOME_REF.finditer(text):
        path = os.path.normpath(HOME + m.group(1))
        if not any(path == p or path.startswith(p + os.sep) for p in ALLOWED_HOME):
            refuse(f"{path} is outside your workspace; work only inside {ARM_DIR}/ws")
    if "/Users/Shared" in text:
        for m in re.finditer(r"/Users/Shared[^\s'\"`;|&)?,:!*\\]*", text):
            p = os.path.normpath(m.group(0))
            if not (p.startswith(ARM_DIR + "/ws") or p.startswith(ARM_DIR + "/cfg/skills")
                    or p.startswith(ARM_DIR + "/cfg/projects") or p == ARM_DIR):
                refuse(f"{p} is outside your workspace; work only inside {ARM_DIR}/ws")


if __name__ == "__main__":
    main()
