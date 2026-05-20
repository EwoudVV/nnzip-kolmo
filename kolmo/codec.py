"""Arithmetic coder wrapper around constriction.

Kept as a thin shim so the rest of the package doesn't import constriction
directly — that makes it easier to swap in a custom (deterministic, portable)
arithmetic coder at Rung 2.
"""

import numpy as np
import constriction


def _categorical(probs: np.ndarray):
    p = probs.astype(np.float64, copy=False)
    return constriction.stream.model.Categorical(p, perfect=False)


class RangeEncoder:
    def __init__(self):
        self._enc = constriction.stream.queue.RangeEncoder()

    def encode(self, symbol: int, probs: np.ndarray) -> None:
        self._enc.encode(symbol, _categorical(probs))

    def encode_uniform(self, symbol: int, n_symbols: int) -> None:
        self.encode(symbol, np.ones(n_symbols, dtype=np.float64) / n_symbols)

    def finish(self) -> bytes:
        return self._enc.get_compressed().tobytes()


class RangeDecoder:
    def __init__(self, data: bytes):
        words = np.frombuffer(data, dtype=np.uint32).copy()
        self._dec = constriction.stream.queue.RangeDecoder(words)

    def decode(self, probs: np.ndarray) -> int:
        return int(self._dec.decode(_categorical(probs)))

    def decode_uniform(self, n_symbols: int) -> int:
        probs = np.ones(n_symbols, dtype=np.float64) / n_symbols
        return self.decode(probs)
