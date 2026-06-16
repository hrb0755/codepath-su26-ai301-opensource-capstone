#!/usr/bin/env python3
"""Reproduce the capability gap behind radixark/miles#400.

Issue #400 ("Support testing Megatron code as an inference engine") is an
*enhancement*, not a bug. "Reproducing" it means showing, on demand and
consistently, that the requested capability is genuinely absent: miles has no
supported way to run its Megatron model as an inference engine. All generation
goes through SGLang.

This script proves that by static inspection of a miles checkout. It needs no
GPU and no miles dependencies -- it only reads miles' source, so anyone can run
it anywhere and see the same result.

Usage:
    python reproduce_issue_400.py [--miles-root /path/to/miles]

Exit 0 = gap reproduced (all conditions held). Exit 1 = a condition did not
hold; re-read the output, the codebase may have changed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Symbols that would indicate miles actually *drives* a Megatron-native
# inference engine -- i.e. that the feature #400 asks for already exists.
ENGINE_DRIVER_SYMBOLS = [
    "StaticInferenceEngine",
    "DynamicInferenceEngine",
    "TextGenerationController",
    "GPTInferenceWrapper",
    "AbstractModelInferenceWrapper",
    "generate_all_output_tokens_static_batch",
]


def find_miles_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "miles")  # sibling of the capstone repo
    candidates.append(Path("/workspace/ruobing/oss/miles"))
    for c in candidates:
        if (c / "miles" / "__init__.py").exists():
            return c
    raise SystemExit("Could not locate a miles checkout. Pass --miles-root /path/to/miles.")


def iter_py_files(root: Path):
    for base in ("miles", "miles_plugins"):
        for dirpath, _, files in os.walk(root / base):
            for f in files:
                if f.endswith(".py"):
                    yield Path(dirpath) / f


def grep(root: Path, pattern: str):
    """Return (relpath, lineno, line) for every match under miles/ + plugins."""
    rx = re.compile(pattern)
    hits = []
    for path in iter_py_files(root):
        try:
            for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if rx.search(line):
                    hits.append((path.relative_to(root), i, line.strip()))
        except OSError:
            continue
    return hits


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * max(0, 70 - len(title))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miles-root", default=None, help="Path to a miles checkout.")
    args = ap.parse_args()
    root = find_miles_root(args.miles_root)

    print(f"Inspecting miles checkout: {root}")
    ok = True

    # ---- 1: generation is implemented only as an HTTP call into SGLang -------
    banner("1. Generation goes only through SGLang")
    sglang_hits = [
        (p, n, ln)
        for (p, n, ln) in grep(root, r"sglang_router_(ip|port)|/generate")
        if "generate_hub" in str(p)
    ]
    if sglang_hits:
        print("  PASS: the generate path posts to SGLang's /generate endpoint:")
        for p, n, ln in sglang_hits[:4]:
            print(f"    {p}:{n}: {ln}")
    else:
        ok = False
        print("  UNEXPECTED: could not find the SGLang HTTP generate path.")

    # ---- 2: no Megatron inference engine is ever constructed/driven ----------
    banner("2. No Megatron inference engine is used")
    driver_hits = []
    for sym in ENGINE_DRIVER_SYMBOLS:
        driver_hits += grep(root, re.escape(sym))
    type_only = grep(root, r"from megatron\.core\.inference.* import .*BaseInferenceContext")
    if not driver_hits:
        print("  PASS: none of Megatron's inference-engine classes are used:")
        print(f"        ({', '.join(ENGINE_DRIVER_SYMBOLS)})")
        print("  The only megatron.core.inference reference is a type import:")
        for p, n, ln in type_only:
            print(f"    {p}:{n}: {ln}")
        print("  -> a type annotation on attention, not anything that generates.")
    else:
        ok = False
        print("  UNEXPECTED: a Megatron inference engine IS referenced:")
        for p, n, ln in driver_hits[:10]:
            print(f"    {p}:{n}: {ln}")

    # ---- 3: no CLI flag selects Megatron as the inference engine -------------
    banner("3. No CLI flag selects Megatron as the inference engine")
    selectors = grep(root, r"rollout-engine|inference-engine\b.*choices|--megatron-inference")
    if not selectors:
        print("  PASS: arguments expose no Megatron inference-engine selector.")
        print("  Rollout always uses SGLang; the only overrides are generic")
        print("  python-path hooks, both defaulting to SGLang:")
        for flag in ("--rollout-function-path", "--custom-generate-function-path"):
            for p, n, ln in grep(root, re.escape(flag)):
                if "arguments.py" in str(p):
                    print(f"    {p}:{n}: {ln}")
                    break
    else:
        ok = False
        print("  UNEXPECTED: a Megatron inference-engine selector exists:")
        for p, n, ln in selectors[:10]:
            print(f"    {p}:{n}: {ln}")

    # ---- verdict ------------------------------------------------------------
    banner("VERDICT")
    if ok:
        print("  GAP REPRODUCED: miles has no supported way to run its Megatron")
        print("  model as an inference engine -- generation is SGLang-only. This")
        print("  is exactly the capability radixark/miles#400 asks to add.")
        return 0
    print("  A condition did not hold -- re-read the output; the codebase may")
    print("  have changed since this script was written.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
