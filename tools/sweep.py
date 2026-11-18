"""Try parameter sets for one agent folder; print its mean rank per field.

  uv run python tools/sweep.py --agent tight_v3 --seeds 6 '[{}, {"open_bid": 4.0}]'

Edits src/dme_simulator/agents/<agent>/agent.py in place for each candidate and
restores the original at the end. All other bundles in agents/ play too.
Fields: big = default + scaled_market with the 22-agent zoo (portal-sized);
builtins = only the bundles in agents/.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "big": ["--config", "config/default.yaml", "--config", "config/scaled_market.yaml",
            "--extra", "sparring_zoo"],
    "nozoo": [],
    "auction": ["--config", "config/tutorial_auction.yaml"],
    # guess of the portal's default_extended: 240 turns, 40k capital, ~400 customers/turn
    "ext": ["--config", "config/local_ext_demand.yaml", "--extra", "sparring_zoo"],
}


def set_params(text: str, params: dict) -> str:
    for key, value in params.items():
        text, n = re.subn(rf"(\n    {key} = )[^\s#]+", rf"\g<1>{value}", text)
        assert n == 1, key
    return text


def ranks(team: str, args: list[str], seeds: int) -> str:
    out = subprocess.run(
        ["uv", "run", "python", "tools/arena.py", "--seeds", str(seeds), *args],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == team:
            return " ".join(parts[1:2] + parts[3:])
    raise RuntimeError(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--agent", default="tight_v3")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--fields", default="big")
    a = ap.parse_args()
    path = ROOT / "src/dme_simulator/agents" / a.agent / "agent.py"
    original = path.read_text()
    try:
        for cand in json.loads(a.candidates):
            path.write_text(set_params(original, cand))
            res = {f: ranks(a.agent, FIELDS[f], a.seeds) for f in a.fields.split(",")}
            print(f"{json.dumps(cand):<75} " + "  ".join(f"{f}: {r}" for f, r in res.items()), flush=True)
    finally:
        path.write_text(original)


if __name__ == "__main__":
    main()
