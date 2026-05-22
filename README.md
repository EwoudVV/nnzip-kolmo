# kolmo

Online-trained neural compression. The model is **grown from scratch during compression** and re-grown identically during decompression, so the trained weights never need to be stored — they live implicitly in the algorithm plus the data.

This is the same architecture that the current Hutter Prize contenders use, in contrast to a pretrained-model approach like [nnzip](https://github.com/EwoudVV/nnzip) (which ships a 250 MB GPT-2 with the tool).

**Name:** the file format is named after [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov). The compressed payload is, in spirit, the shortest program known to us that reproduces the input — which is exactly what Kolmogorov complexity asks for.

## Status: Rung 2 — cross-machine deterministic

| Rung | Goal | Status |
|---:|---|---|
| 1 | PyTorch online-training prototype, single machine | ✅ done |
| 2 | Bit-deterministic on a single machine (drop PyTorch) | ✅ done — `KOLMO_FIXED=1` |
| 3 | Beat nnzip's compression ratio on long files | — |
| 4 | Cross-platform fixed-point math (Mac/Linux/x86/ARM identical) | ✅ done (folded into Rung 2 via Q15 integer engine; CI verifies on every push) |
| 5 | Match SOTA on enwik9 (~0.85 bpb) | — |
| 6 | Submit to Marcus Hutter and win the actual prize | — |

Rung 2 collapsed Rung 4 into itself: the same Q15 fixed-point engine that drops PyTorch's nondeterminism also gives byte-identical output across Mac/Linux/x86/ARM, because integer addition is associative regardless of SIMD width or thread count. See [`benchmarks/determinism/README.md`](benchmarks/determinism/README.md) for the acceptance test and CI workflow.

Three benchmarks tell different parts of the story.

**Mixed local corpus** (8.9 KB of prose / wiki / dialogue / markdown) with the deterministic 5.5 KB English seed warmup baked into the decompressor source:

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 1 KB | 55.3% | **46.9%** |
| 2 KB | 50.0% | **44.9%** |
| 4 KB | 47.8% | **44.3%** |
| 8 KB | 45.7% | **43.9%** |

**Procedurally extended long corpus** (clean local prose + deterministic generated paragraphs, used to test the long-file slope):

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 8 KB | 45.7% | **43.9%** |
| 16 KB | 35.5% | 35.6% |
| 32 KB | 20.0% | **19.5%** |

**Real-prose corpus** (20 KB cleaned Pride and Prejudice — the honest test, text we never tuned against):

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 1 KB | 27.4% | **26.6%** |
| 2 KB | 23.4% | 23.4% |
| 4 KB | 23.6% | 24.0% |
| 8 KB | 27.4% | 27.7% |
| 16 KB | 34.5% | 35.1% |

The mixed local corpus has more vocabulary diversity than P&P's narrow narrative voice, so gzip's static codebook becomes less efficient there and kolmo's online learning pulls ahead. The procedural extension is heavy on structural repetition that COPY_MAX=256 captures in single copy events. On uniform real prose, kolmo is **competitive with gzip** but doesn't dominate — its learning advantage roughly cancels gzip's optimized LZ77.

This is the Rung 1 milestone: an online-trained neural compressor that runs end-to-end, ties or beats gzip on most regimes, and exposes the underlying tradeoff clearly. Architectural pieces that got it here: deterministic seed warmup, decoupled copy lookup window (8 KB), adaptive distributions for offset / length / event-flag, raising COPY_MAX from 32 → 256 so long structural matches collapse into single copy events.

## How it works (conceptually)

1. Both compressor and decompressor start a transformer from the same fixed RNG seed.
2. For each byte in the input:
   - Run a forward pass to get a probability distribution over the next byte
   - Use that distribution to encode the actual next byte with arithmetic coding
   - Run a backward pass on the actual byte (so the model learns from it)
3. If the next bytes exactly match recent history, encode a small copy event instead of literal bytes.
4. The compressed file contains *only* the arithmetic-coded bitstream — no model weights.
5. Decompression mirrors the loop exactly: same starting weights, same training schedule, same probability distributions at every step.

## Development

```sh
git clone https://github.com/EwoudVV/nnzip-kolmo
cd nnzip-kolmo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

## License

MIT
