"""Ratio-regression guard for the determinism CI.

Why this exists:

The other probes in this folder (hash_fixed_compress, hash_fixed_compress_long,
etc.) check that the *bits* of the output blob agree across machines. That
catches platform-dependent bugs — but it does NOT catch the case where a
refactor leaves cross-platform consistency intact while silently making the
blob 5 % larger on every platform equally.

This probe is the complementary check: hold the compressor's *ratio*
honest. It runs a fixed payload through `compress()` in the same fixed-
point + skip-prime configuration the other probes use, then asserts the
output is no larger than a known good floor. If a refactor regresses the
ratio, this probe fails immediately on every runner with a clear error,
*before* the determinism comparison step would even start.

Bumping the bound:
    When a legitimate ratio improvement lands, run the probe locally,
    note the new `len(blob)`, and update `EXPECTED_MAX_BYTES` below to
    the new value + RATIO_REGRESSION_MARGIN_BYTES. The margin gives small
    benign drifts (e.g. an Adam-step reorder that shifts ratio by ±1 B)
    room without paging the maintainer.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

os.environ["KOLMO_FIXED"] = "1"
os.environ["KOLMO_SKIP_PRIME"] = "1"
# Same defaults as hash_fixed_compress_long — this probe must use the
# CURRENT shipped defaults (PPM + adaptive + order-3 by the time of
# writing) so it actually measures the production ratio path.
os.environ.setdefault("KOLMO_MODEL", "full")
os.environ.setdefault("KOLMO_LITERAL", "ppm")

from kolmo import compress, decompress  # noqa: E402


# The fixed payload is the same as hash_fixed_compress_long so we only
# need to maintain one canonical "longer probe" payload.
PAYLOAD = (
    b"The quick brown fox jumps over the lazy dog. "
    b"The quick brown fox jumps over the lazy dog again. "
    b"== Section ==\n"
    b"[[link|text]] {{template}}\n"
    b"Numbers: 1, 2, 3, 1024, 65536, 3.14159.\n"
    b"Determinism across machines, byte for byte, is the goal.\n"
)

# Floor + margin. The floor is the actual measured `len(blob)` when this
# probe was last touched; the margin absorbs ±benign drifts. If the probe
# fires, EITHER fix the ratio regression OR (if you're sure the new
# behavior is correct and an improvement landed) bump these numbers.
EXPECTED_MAX_BYTES = 175           # current value: see commit message
RATIO_REGRESSION_MARGIN_BYTES = 3  # small benign drift allowed


def main() -> None:
    blob = compress(PAYLOAD)
    out = decompress(blob)
    assert out == PAYLOAD, "fixed-mode compress/decompress mismatch on this machine"

    print(f"input bytes: {len(PAYLOAD)}")
    print(f"output bytes: {len(blob)}")
    print(f"output sha256: {hashlib.sha256(blob).hexdigest()}")
    print(
        f"ratio floor: <= {EXPECTED_MAX_BYTES + RATIO_REGRESSION_MARGIN_BYTES} B"
    )

    if len(blob) > EXPECTED_MAX_BYTES + RATIO_REGRESSION_MARGIN_BYTES:
        print(
            f"\nRATIO REGRESSION: blob is {len(blob)} B, expected "
            f"<= {EXPECTED_MAX_BYTES} (+ {RATIO_REGRESSION_MARGIN_BYTES} B margin). "
            f"Either fix the regression, or — if the larger output is a "
            f"correct outcome of an intended algorithm change — bump "
            f"EXPECTED_MAX_BYTES in this file to the new measured value.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
