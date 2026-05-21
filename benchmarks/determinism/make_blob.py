"""Make a known blob, save to disk."""
import os, sys
mac_repo = "/Users/kids/Documents/nnzip-kolmo"
if os.path.exists(mac_repo) and mac_repo not in sys.path:
    sys.path.insert(0, mac_repo)
from kolmo import compress
DATA = b"The quick brown fox jumps over the lazy dog. " * 50
open(sys.argv[1], "wb").write(compress(DATA))
print(f"wrote {sys.argv[1]}: input {len(DATA)}B")
