"""Regression tests for the adaptive training cadence."""

from kolmo._engine import CONTEXT, training_block_size_at


def test_training_block_size_never_consumes_whole_context():
    """A training slice needs one preceding history token.

    The long-file schedule used to grow to 256/512 bytes while CONTEXT was
    256. At 32KB this produced an empty logits slice: there were target bytes
    but no preceding token left in the training window to predict the first
    one. Keep the schedule capped at CONTEXT-1 so long files do not crash.
    """
    for observed in (0, 4096, 8192, 16384, 32768, 1_000_000_000):
        assert training_block_size_at(observed) <= CONTEXT - 1
