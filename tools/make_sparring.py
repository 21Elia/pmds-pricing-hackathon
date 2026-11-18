"""Write sparring copies of an agent with different class parameters.

  uv run python tools/make_sparring.py            # variants of tight -> sparring/
Each variant is a full copy (one concrete class per bundle), with the listed
class attributes replaced. Folder names become team IDs.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/dme_simulator/agents/tight/agent.py"
OUT = ROOT / "sparring"

VARIANTS = {
    "sp_maxbid": {"value_share": 1.5, "bid_start": 2.5, "bid_up": 2.0, "bid_down": 0.97},
    "sp_softbid": {"value_share": 0.5, "bid_up": 1.25},
    "sp_cheaper": {"ladder_low": 0.45, "ladder_high": 0.8},
    "sp_dearer": {"ladder_low": 0.8, "ladder_high": 1.3},
    "sp_bighoard": {"min_stock": 12, "max_stock": 60, "cover_turns": 6.0},
}


def main() -> None:
    src = SOURCE.read_text()
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, params in VARIANTS.items():
        text = src
        for key, value in params.items():
            text, n = re.subn(rf"(\n    {key} = )[^\s#]+", rf"\g<1>{value}", text)
            assert n == 1, (name, key)
        (OUT / name).mkdir(parents=True)
        (OUT / name / "agent.py").write_text(text)
    print("written:", ", ".join(VARIANTS))


if __name__ == "__main__":
    main()
