#!/usr/bin/env python3
"""PreToolUse guard for the KDA matrix arms.

Refuses a tool call whose input mentions
  - the reference implementation or the contest winners' code (repositories and URLs),
  - the black-box reference tool's own files (it may be run, not read),
  - any path in this machine's home directory outside the arm's own workspace and skills,
  - other sessions' scratch directories.
Exit code 2 with a reason on stderr is how Claude Code is told to refuse the call.
"""
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
ARM_DIR = os.environ.get("KDA_MX_ARM_DIR", "")

FORBIDDEN_TEXT = [
    r"mlsys2026-flashinfer-contest",       # KDA 0.1 submission and the KDA-0.5 release
    r"kda0\.5", r"humanfia/kda", r"kda-for-kda", r"NVlabs/kda/tree",
    r"kachua-mlsys", r"flashinfer-challenge", r"DeepSeek-Sparse-Attention-Kernels",
    r"\.kda05ref", r"kda05_ref",
    r"/(private/)?tmp/claude-",
]
# FA4-backward arms only: flash-attn-4 is their baseline, run but not read, and /Users/Shared holds
# another FA4-backward experiment's workspace.
FA4_ARM = "-fa4-" in os.path.basename(ARM_DIR)
FA4_FORBIDDEN_TEXT = [
    r"site-packages/flash_attn", r"flash_attn/cute", r"flash_attn_4", r"flash-attn-4[^\s'\"]*\.(tar|whl|zip)",
    r"Dao-AILab", r"github\.com/[^\s'\"]*flash-?attention", r"inspect\.getsource", r"getsourcelines",
    r"pip\s+download[^|;&\n]*flash", r"unzip[^|;&\n]*flash", r"/Users/Shared",
    r"flash_attn.{0,300}__file__", r"__file__.{0,300}flash_attn", r"\bfind\b[^|;&\n]*flash_attn",
]
READ_TOOL = (r"\b(cat|less|more|head|tail|sed|awk|vi|vim|nano|strings|xxd|od|cp|scp|rsync|tar|zip|"
             r"grep|rg|base64|file|python[0-9.]*\s+-c|bash\s+-x)\b[^|;&\n]*kda05-ref")
ALLOWED_HOME = [
    os.path.join(ARM_DIR, "ws"),
    os.path.join(ARM_DIR, "cfg", "skills"),
    os.path.join(HOME, ".cybertronctl"),
    os.path.join(HOME, ".tsh"),
    os.path.join(HOME, ".local", "bin"),
    os.path.join(HOME, ".cache"),
    os.path.join(HOME, "Library", "Caches"),
]
HOME_REF = re.compile(r"(?:" + re.escape(HOME) + r"|~(?=/)|\$HOME\b|\$\{HOME\})([^\s'\"`;|&)?,:!*\\]*)")
BARE_HOME = re.compile(r"\bcd(\s+(~|\$HOME|\$\{HOME\}))?\s*(?=$|[;&|'\"]|\\n)")


def refuse(why):
    sys.stderr.write(f"Refused by the experiment's guard: {why}\n")
    sys.exit(2)


def main():
    try:
        call = json.load(sys.stdin)
    except Exception:
        return
    text = json.dumps(call.get("tool_input", {}), ensure_ascii=False)
    text = text.replace("\\/", "/")
    for pat in FORBIDDEN_TEXT:
        if re.search(pat, text, re.IGNORECASE):
            refuse("this touches the reference implementation, contest submissions, or files that are off limits")
    if FA4_ARM:
        inp = dict(call.get("tool_input", {}) or {})
        if call.get("tool_name", "") in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            inp = {k: v for k, v in inp.items() if k in ("file_path", "notebook_path")}
        fa4_text = json.dumps(inp, ensure_ascii=False).replace("\\/", "/")
        for pat in FA4_FORBIDDEN_TEXT:
            if re.search(pat, fa4_text, re.IGNORECASE):
                refuse("flash-attn's source code and other experiments' files are off limits "
                       "(the flash-attn-4 baseline may be run, not read)")
    if re.search(READ_TOOL, text):
        refuse("kda05-ref may be run, but its implementation may not be read")
    if not ARM_DIR:
        return
    if BARE_HOME.search(text):
        refuse(f"the home directory is outside your workspace; work only inside {os.path.join(ARM_DIR, 'ws')}")
    for m in HOME_REF.finditer(text):
        path = HOME + m.group(1)
        path = os.path.normpath(path)
        if not any(path == p or path.startswith(p + os.sep) for p in ALLOWED_HOME):
            refuse(f"{path} is outside your workspace; work only inside {os.path.join(ARM_DIR, 'ws')}")


if __name__ == "__main__":
    main()
