"""Try to decompress a blob made elsewhere."""
import os, sys
mac_repo = "/Users/kids/Documents/nnzip-kolmo"
if os.path.exists(mac_repo) and mac_repo not in sys.path:
    sys.path.insert(0, mac_repo)
from kolmo import decompress
blob = open(sys.argv[1], "rb").read()
EXPECTED = b"The quick brown fox jumps over the lazy dog. " * 50
try:
    out = decompress(blob)
except Exception as e:
    print(f"DECOMPRESS THREW: {type(e).__name__}: {e}")
    sys.exit(1)
if out == EXPECTED:
    print(f"ROUND-TRIP OK: {len(out)}B recovered, matches expected")
else:
    matching_prefix = 0
    for a, b in zip(out, EXPECTED):
        if a == b:
            matching_prefix += 1
        else:
            break
    print(f"DECOMPRESS RAN but output DIFFERS")
    print(f"  output: {len(out)}B, expected: {len(EXPECTED)}B")
    print(f"  matching prefix: {matching_prefix} bytes")
    if matching_prefix < 30:
        print(f"  output[:60]: {out[:60]!r}")
        print(f"  expected[:60]: {EXPECTED[:60]!r}")
