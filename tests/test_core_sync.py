"""src/core must be an unedited copy of lisaClust's core (tools/sync_core.sh)."""

import hashlib
from pathlib import Path

CORE = Path(__file__).parents[1] / "src" / "core"


def test_core_matches_checksums():
    lines = [l for l in (CORE / "CHECKSUMS").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((CORE / name).read_bytes()).hexdigest() == digest, name
