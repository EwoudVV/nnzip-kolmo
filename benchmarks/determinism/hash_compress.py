"""Compress a known input, hash the output. Run on any machine."""
import hashlib
import os
import sys

# Mac: kolmo not pip-installed, point at repo. Windows: pip-installed already.
mac_repo = "/Users/kids/Documents/nnzip-kolmo"
if os.path.exists(mac_repo) and mac_repo not in sys.path:
    sys.path.insert(0, mac_repo)

from kolmo import compress

DATA = b"The quick brown fox jumps over the lazy dog. " * 50  # ~2.2 KB
blob = compress(DATA)
print(f"input bytes: {len(DATA)}")
print(f"output bytes: {len(blob)}")
print(f"output sha256: {hashlib.sha256(blob).hexdigest()}")
print(f"first 32 bytes: {blob[:32].hex()}")
