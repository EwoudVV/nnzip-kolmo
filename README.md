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

Current prototype result on a clean 8.9 KB mixed text corpus, with a deterministic 5.5 KB English seed warmup compiled into the decompressor source:

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 1 KB | 55.3% | **46.9%** |
| 2 KB | 50.0% | **44.9%** |
| 4 KB | 47.8% | **44.3%** |
| 8 KB | 45.7% | **43.9%** |

The 1 KB result is seed-dominated, so the cleaner milestone is the 2 KB/4 KB/8 KB win. The copy mechanism looks back 8 KB independently of the transformer's 256-byte context, with three adaptive models (offset, length, event-flag) that learn the corpus's repetition structure online. Match lengths run up to 128 bytes.

Longer slope test on a deterministic 32 KB mixed corpus:

| Size | gzip -9 | kolmo |
|---:|---:|---:|
| 8 KB | 45.7% | **43.9%** |
| 16 KB | **35.5%** | 35.7% |
| 32 KB | 20.0% | **19.7%** |

**kolmo beats gzip across 1 KB–8 KB and at 32 KB**, with a 0.2 pp gap at 16 KB. This is the Rung 1 milestone: a from-scratch online-trained neural compressor that matches a tuned classical compressor across two orders of magnitude of file size. Architectural changes that got us here: deterministic seed warmup, decoupled copy lookup window, adaptive distributions for offset / length / event-flag, and raising COPY_MAX so long structural matches collapse into single copy events.

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
