"""kolmo: online-trained neural compression.

The model is initialized from a fixed seed, then trained byte-by-byte during
compression. The decompressor uses the same seed and the same training schedule,
so both sides produce identical weights at every step. The compressed file
contains the arithmetic-coded probability stream — never the model itself.
"""

__version__ = "0.0.1"

from kolmo.compress import compress
from kolmo.decompress import decompress
from kolmo.model import KolmoTransformer
from kolmo.tokenizer import ByteTokenizer

__all__ = [
    "compress",
    "decompress",
    "KolmoTransformer",
    "ByteTokenizer",
    "__version__",
]
