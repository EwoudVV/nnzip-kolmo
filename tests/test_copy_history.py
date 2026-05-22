"""Copy-window mechanics for long files.

The neural model dominates small-file runtime, but on large files the copy
machinery must not grow memory or copy work with total file length. Only the
last COPY_WINDOW bytes can ever be addressed by an offset, so the history buffer
should stay bounded and find_copy should operate directly on that bounded
bytearray instead of copying the whole file prefix at every position.
"""

from kolmo._engine import (
    COPY_MIN,
    COPY_WINDOW,
    RollingCopyMatcher,
    append_copy_history,
    find_copy,
)


def test_find_copy_accepts_bytearray_history():
    """find_copy should operate on bytearray history directly.

    This prevents compress() from doing bytes(copy_history) at every byte
    position, which was O(total history) work per probe on large files.
    """
    known = bytearray(b"abcdefgh")
    data = b"abcdefghZ"
    assert find_copy(data, 0, known) == (8, 8)


def test_find_copy_ignores_history_older_than_copy_window():
    old = b"abcdefgh"
    known = bytearray(old + (b"x" * COPY_WINDOW))
    data = old + b"Z"
    assert find_copy(data, 0, known) is None


def test_append_copy_history_preserves_recent_tail_and_bounds_memory():
    history = bytearray()
    old = b"ABCDEFGH"
    for value in old:
        append_copy_history(history, value)
    for value in b"x" * (3 * COPY_WINDOW):
        append_copy_history(history, value)

    assert len(history) <= 2 * COPY_WINDOW
    assert bytes(history[-COPY_MIN:]) == b"x" * COPY_MIN
    # The oldest bytes have been trimmed, so the first COPY_MIN-byte pattern
    # cannot be found any more even though it existed in the original stream.
    assert find_copy(old + b"Z", 0, history) is None


def test_rolling_copy_matcher_finds_repeat():
    data = b"abcdefghZZabcdefgh"
    matcher = RollingCopyMatcher(data)
    assert matcher.find(10) == (10, 8)


def test_rolling_copy_matcher_respects_window():
    data = b"abcdefgh" + (b"x" * 16) + b"abcdefgh"
    matcher = RollingCopyMatcher(data, window=16)
    assert matcher.find(24) is None


def test_rolling_copy_matcher_uses_non_overlapping_matches():
    data = b"abcdefghabcdefghabcdefgh"
    matcher = RollingCopyMatcher(data)
    # The previous match is 8 bytes behind, so even though more bytes match
    # after that, non-overlap caps the copy length at offset=8.
    assert matcher.find(8) == (8, 8)


def test_rolling_copy_matcher_index_stays_bounded():
    data = bytes((i * 131 + 17) % 251 for i in range((3 * COPY_WINDOW) + 1024))
    matcher = RollingCopyMatcher(data)
    for pos in range(COPY_MIN, len(data), 257):
        matcher.find(pos)

    indexed = sum(len(candidates) for candidates in matcher._index.values())
    assert indexed <= COPY_WINDOW
