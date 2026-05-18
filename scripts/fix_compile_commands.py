#!/usr/bin/env python3
"""Post-processes build/compile_commands.json into a clangd-friendly version.

For nvcc (.cu) entries:
  - Expands --options-file RSP files inline (clangd cannot read them)
  - Strips nvcc-only flags that clang++ rejects
  - Replaces nvcc with clang++
  - Injects -x cuda, --cuda-path, and CUDA include paths

Plain C++ entries are passed through unchanged.
Run automatically by the CMake 'clangd_db' target after every build.
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

_STRIP_EXACT = frozenset([
    "-forward-unknown-to-host-compiler",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    "--lineinfo",
])

_STRIP_PREFIX = (
    "-ccbin=",               # nvcc: host compiler path (value glued with =)
    "--compiler-bindir=",
    "--compiler-options=",
    "--generate-code=",
    "--rdc=",
    "-rdc=",
    "--relocatable-device-code=",
    "--gpu-architecture=",
    "--gpu-code=",
)

_STRIP_PAIR = frozenset(["-ccbin", "--compiler-bindir", "-x"])


def expand_rsp(token, cwd):
    p = Path(token) if Path(token).is_absolute() else Path(cwd) / token
    if p.exists():
        return shlex.split(p.read_text())
    return []


def filter_tokens(tokens, cwd):
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--options-file" and i + 1 < len(tokens):
            rsp = expand_rsp(tokens[i + 1], cwd)
            out.extend(filter_tokens(rsp, cwd))
            i += 2
            continue
        if t in _STRIP_PAIR and i + 1 < len(tokens):
            i += 2
            continue
        if t in _STRIP_EXACT:
            i += 1
            continue
        if any(t.startswith(p) for p in _STRIP_PREFIX):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def fix_cuda_entry(entry, cuda_path, sm, extra_includes):
    tokens = shlex.split(entry["command"])
    filtered = filter_tokens(tokens[1:], entry["directory"])
    cmd = ["clang++"] + filtered + [
        "-x", "cuda",
        "--cuda-gpu-arch=sm_" + sm,
        "--cuda-path=" + cuda_path,
        "-I" + cuda_path + "/include",
        "-I" + cuda_path + "/targets/x86_64-linux/include",
        "-Wno-unknown-cuda-version",
        "-Wno-unused-command-line-argument",
    ]
    for inc in extra_includes:
        flag = "-I" + inc
        if flag not in cmd:
            cmd.append(flag)
    result = dict(entry)
    result["command"] = shlex.join(cmd)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cuda-path", default="/opt/cuda")
    ap.add_argument("--sm", default="86")
    ap.add_argument("--include", action="append", default=[], dest="includes")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        sys.stderr.write("[fix_compile_commands] " + str(src) + " not found, skipping.\n")
        sys.exit(0)

    db = json.loads(src.read_text())
    fixed = []
    n_cu = 0
    n_cpp = 0

    for entry in db:
        compiler = shlex.split(entry["command"])[0]
        if "nvcc" in Path(compiler).name:
            fixed.append(fix_cuda_entry(entry, args.cuda_path, args.sm, args.includes))
            n_cu += 1
        else:
            fixed.append(entry)
            n_cpp += 1

    if dst.is_symlink():
        dst.unlink()

    with open(str(dst), "w") as f:
        f.write(json.dumps(fixed, indent=2))
        f.write("\n")

    sys.stdout.write("[fix_compile_commands] wrote " + str(n_cu) + " CUDA + " + str(n_cpp) + " C++ entries -> " + str(dst) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
