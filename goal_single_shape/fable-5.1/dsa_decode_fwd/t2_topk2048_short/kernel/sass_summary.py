"""Summarize cuobjdump -sass output (stdin) for the main kernel: size, opcode histogram, math region."""
import collections
import re
import sys

funcs, cur = {}, None
for line in sys.stdin:
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
        cur = m.group(1)
        funcs[cur] = []
        continue
    m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(@!?U?P\d\s+)?([A-Z0-9_.]+)", line)
    if m and cur is not None:
        funcs[cur].append(m.group(2))
target = sys.argv[2] if len(sys.argv) > 2 else "dsa_sparse_attn_kernel"
for name, ops in funcs.items():
    if target not in name:
        continue
    base = collections.Counter(op.split(".")[0] for op in ops)
    keys = ["HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "UTMALDG", "SYNCS",
            "LDG", "STG", "LDS", "STS", "SHFL", "VOTE", "MUFU", "FFMA", "FMUL", "FADD", "BAR", "BRA", "BSSY", "BSYNC",
            "EXIT", "S2R", "IMAD", "FSEL", "FMNMX", "LOP3", "SHF", "PRMT", "F2FP", "MOV", "ISETP", "POPC"]
    print(f"   SASS {sys.argv[1]}: {len(ops)} instructions ({len(ops) * 16} B); " +
          " ".join(f"{k}={base[k]}" for k in keys if base[k]))
    idx = [i for i, op in enumerate(ops) if op.startswith("SHFL")]
    if idx and "--region" in sys.argv:
        lo, hi = max(0, idx[0] - 4), min(len(ops), idx[-1] + 5)
        seq = [op.split(".")[0] for op in ops[lo:hi]]
        rl, prev, n = [], None, 0
        for op in seq + [None]:
            if op == prev:
                n += 1
            else:
                if prev is not None:
                    rl.append(f"{prev}x{n}" if n > 1 else prev)
                prev, n = op, 1
        print(f"   math region [{lo},{hi}): " + " ".join(rl))
