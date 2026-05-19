"""Byte-level tokenizer.

The Hutter Prize scores raw bytes, so we don't want a BPE that might split or
merge characters in ways that depend on the input. Each of the 256 possible
byte values is its own token. Trivial, but having it as a named module keeps
the door open for later experiments (e.g. a learned BPE for non-Hutter use).
"""


class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, data: bytes) -> list[int]:
        return list(data)

    def decode(self, tokens: list[int]) -> bytes:
        return bytes(tokens)
