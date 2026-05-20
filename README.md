# kolmo

Online-trained neural compression. The model is **grown from scratch during compression** and re-grown identically during decompression, so the trained weights never need to be stored — they live implicitly in the algorithm plus the data.

This is the same architecture that the current Hutter Prize contenders use, in contrast to a pretrained-model approach like [nnzip](https://github.com/EwoudVV/nnzip) (which ships a 250 MB GPT-2 with the tool).

**Name:** the file format is named after [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov). The compressed payload is, in spirit, the shortest program known to us that reproduces the input — which is exactly what Kolmogorov complexity asks for.

## Status: Rung 1 — PyTorch prototype

| Rung | Goal | Status |
|---:|---|---|
| 1 | PyTorch online-training prototype, single machine | **in progress** |
| 2 | Bit-deterministic on a single machine (drop PyTorch) | — |
| 3 | Beat nnzip's compression ratio on long files | — |
| 4 | Cross-platform fixed-point math (Mac/Linux/x86/ARM identical) | — |
| 5 | Match SOTA on enwik9 (~0.85 bpb) | — |
| 6 | Submit to Marcus Hutter and win the actual prize | — |

Current prototype result on a clean 8.9 KB mixed text corpus:

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 1 KB | 55.3% | **51.6%** |
| 2 KB | 50.0% | **49.4%** |
| 4 KB | **47.8%** | 48.6% |
| 8 KB | **45.7%** | 49.3% |

Kolmo now wins on tiny text prefixes after a deterministic 432-byte seed-corpus warmup. Gzip still wins on longer files because exact substring reuse remains the hard part.

## How it works (conceptually)

1. Both compressor and decompressor start a transformer from the same fixed RNG seed.
2. For each byte in the input:
   - Run a forward pass to get a probability distribution over the next byte
   - Use that distribution to encode the actual next byte with arithmetic coding
   - Run a backward pass on the actual byte (so the model learns from it)
3. The compressed file contains *only* the arithmetic-coded bitstream — no model weights.
4. Decompression mirrors the loop exactly: same starting weights, same training schedule, same probability distributions at every step.

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
