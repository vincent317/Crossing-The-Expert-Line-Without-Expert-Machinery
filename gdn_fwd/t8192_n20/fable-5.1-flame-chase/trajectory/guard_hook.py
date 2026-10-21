#!/usr/bin/env python3
"""PreToolUse guard for the GDN prefill flame-chase arm (1020).

Refuses a tool call that
  - reads or fetches a reference implementation's source: the KDA-0.5 kernels behind the black-box
    kda05-ref tool (and their public repositories), or FlashInfer's GDN prefill kernels,
  - touches any path in this machine's home directory outside the allowed tool/cache dirs,
  - touches /Users/Shared outside the arm's workspace and skills, or other sessions' scratch dirs.
Write/Edit content is not scanned (comments may name the references); only their target path is.
Exit code 2 with a reason on stderr is how Claude Code is told to refuse the call.
"""
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
ARM_DIR = "/Users/Shared/gdn-fwd-fc-1020"
OWN_SCRATCH = "-Users-Shared-gdn-fwd-fc-1020-ws"

FORBIDDEN_TEXT = [
    # KDA-0.5 reference (run only through kda05-ref) and its public homes
    r"\.kda05ref", r"kda05_ref", r"mlsys2026-flashinfer-contest", r"kda0\.5", r"humanfia", r"kda-for-kda",
    r"NVlabs/kda", r"kachua", r"flashinfer-challenge",
    # FlashInfer GDN kernels (a measured reference candidate)
    r"flashinfer/gdn", r"gdn_prefill\.py", r"gdn_kernels", r"gdn_blackwell", r"\.gdnrefs",
    r"github\.com/flashinfer-ai/flashinfer/(blob|tree|raw)[^\s'\"]*gdn", r"raw\.githubusercontent\.com/flashinfer-ai",
    r"flashinfer[^\n]*(__file__|__path__|find_spec|\.origin)", r"site-packages/flashinfer/(gdn|.*gdn)",
    r"inspect\.getsource", r"getsourcelines", r"__code__", r"\bdis\.dis\b",
]
READ_TOOL = (r"\b(cat|less|more|head|tail|sed|awk|vi|vim|nano|strings|xxd|od|cp|scp|rsync|tar|zip|"
             r"grep|rg|base64|file|python[0-9.]*\s+-c|bash\s+-x|sh\s+-x|type|which|readlink)\b[^|;&\n]*kda05-ref")
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
            refuse("reference implementations' source code is off limits (kda05-ref and FlashInfer's GDN may be run, not read)")
    if re.search(READ_TOOL, text):
        refuse("kda05-ref may be run, but its implementation may not be read")
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
