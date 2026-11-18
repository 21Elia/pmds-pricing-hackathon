"""Try parameter sets for tight in several fields; print tight's mean rank per field.

Edits tight/agent.py in place for each candidate and restores the original at the end.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "src/dme_simulator/agents/tight/agent.py"
FIELDS = {
    "builtins": [],
    "fixed": ["--extra", "sparring_fixed"],
    "full": ["--extra", "sparring", "--extra", "sparring_fixed"],
}


def set_params(text: str, params: dict) -> str:
    for key, value in params.items():
        text, n = re.subn(rf"(\n    {key} = )[^\s#]+", rf"\g<1>{value}", text)
        assert n == 1, key
    return text


def tight_rank(args: list[str], seeds: int) -> float:
    out = subprocess.run(
        ["uv", "run", "python", "tools/arena.py", "--seeds", str(seeds), *args],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("tight "):
            return float(line.split()[1])
    raise RuntimeError(out)


def main() -> None:
    candidates = json.loads(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    original = AGENT.read_text()
    try:
        for cand in candidates:
            AGENT.write_text(set_params(original, cand))
            ranks = {f: tight_rank(a, seeds) for f, a in FIELDS.items()}
            print(f"{json.dumps(cand):<70} " + "  ".join(f"{f}={r:.2f}" for f, r in ranks.items()), flush=True)
    finally:
        AGENT.write_text(original)


if __name__ == "__main__":
    main()
