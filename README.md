# kolmo

Online-trained neural compression. The model is **grown from scratch during compression** and re-grown identically during decompression, so the trained weights never need to be stored — they live implicitly in the algorithm plus the data.

This is the same architecture that the current Hutter Prize contenders use, in contrast to a pretrained-model approach like [nnzip](https://github.com/EwoudVV/nnzip) (which ships a 250 MB GPT-2 with the tool).

**Name:** the file format is named after [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov). The compressed payload is, in spirit, the shortest program known to us that reproduces the input — which is exactly what Kolmogorov complexity asks for.

## Status

| Rung | Goal | Status |
|---:|---|---|
| 1 | PyTorch online-training prototype, single machine | ✅ done |
| 2 | Bit-deterministic on a single machine (drop PyTorch's nondeterminism) | ✅ done — `KOLMO_FIXED=1` |
| 3 | Beat gzip on real enwik prefixes, then chase nnzip / Hutter-scale ratios | ✅ started — beats `gzip -9` on 16-128 KB enwik9 prefixes |
| 4 | Cross-platform fixed-point math (Mac/Linux/x86/ARM identical) | ✅ done (folded into Rung 2 via Q15 integer engine; CI verifies on every push) |
| 5 | Match SOTA on enwik9 (~0.85 bpb) | — |
| 6 | Submit to Marcus Hutter and win the actual prize | — |

Rungs 2 and 4 collapsed into one fix: a Q15 fixed-point integer engine (forward, backward, Adam, KV cache, training step). Integer addition is associative regardless of SIMD width or thread count, so the same input produces byte-identical output on Mac, Linux, x86, and ARM. CI on every push runs four hash probes on all three OSes and fails if any runner's transcript drifts from the others.

## Two modes

`kolmo` ships two arithmetic backends behind one API:

| Backend | Trigger | Speed | Cross-machine | Use when |
|---|---|---|---|---|
| PyTorch float32 | default | ~14 ms/byte (inner loop) | ❌ Mac ≠ Windows ≠ CUDA | iterating locally, training experiments |
| Q15 fixed-point | `KOLMO_FIXED=1` | ~285 ms/byte (inner loop) | ✅ identical everywhere | making a blob that must round-trip somewhere else |

The two backends produce *different* blobs even on the same input (they're computing different probabilities under the hood) — but each backend is internally consistent, so a PyTorch-mode blob decompresses with a PyTorch-mode kolmo, and a fixed-mode blob with a fixed-mode kolmo. The blob can't be told apart by file format; the same `MAGIC` header is used.

### Why the fixed-point mode is slower

The remaining gap is honest cost of doing transformer math in mostly Python/numpy integer code instead of vectorized float32 kernels. Some hotspots now use safe accelerators (float64 BLAS for Q15 matmul, numba for integer kernels), but a real enwik9 run still needs more kernel work.

### Pre-computed seed cache

The seed corpus warms the model up before user data arrives — about 5 KB of diverse English / Wikipedia patterns baked into the algorithm. In PyTorch mode this primes in seconds. In fixed mode it takes several minutes (one ~2 s training step per 16 bytes of seed).

But the result is *deterministic*. Same inputs → same primed state, every machine, every run. So `kolmo` saves it to `~/.cache/kolmo/seed_state_<hash>.npz` after the first run. Subsequent runs load it in ~2 seconds.

```sh
# First fixed-mode compress on this machine: 3+ minutes of priming
KOLMO_FIXED=1 kolmo c some-file.txt

# Every subsequent fixed-mode compress: ~2 seconds startup
KOLMO_FIXED=1 kolmo c another-file.txt
```

The cache invalidates automatically when any of (seed corpus, model architecture, init seed, block size, format version) changes — the file name embeds a hash of all of them. Bypass with `KOLMO_NO_SEED_CACHE=1` to force re-prime. Override location with `KOLMO_CACHE_DIR=...`.

## How it works (conceptually)

1. Both compressor and decompressor start a transformer from the **same fixed RNG seed** (a tiny SplitMix64 PRNG, not PyTorch's `manual_seed`, so initial weights are byte-identical on every platform).
2. They both warm up on a **5 KB seed corpus** baked into the source code. The seed costs zero bytes in the output blob — it's part of the algorithm, like a lookup table.
3. For each byte in the input:
   - Run a forward pass to get a probability distribution over the next byte.
   - Mix that neural distribution with an adaptive byte-context side model (dense order-2 + small order-1/order-0 backoff).
   - Encode the actual next byte under that distribution using arithmetic coding — fewer bits when the model predicts correctly.
   - Append the byte to the running history.
4. Every 16 bytes (`BLOCK_SIZE`), do **one backward pass + Adam step** so the model adapts to the file it's currently compressing.
5. If the next 8+ bytes happen to match recent history, encode a small `(offset, length)` copy event instead of running each byte through the neural model — a classic LZ-style fallback for structural repetition that the model would waste bits on.
6. The compressed file contains *only* the arithmetic-coded bitstream + a 4-byte magic + a length header. No model weights.
7. Decompression mirrors the loop exactly: same starting weights, same warmup, same training schedule, same byte-by-byte probability distributions. Both sides walk identical trajectories.

The whole system collapses if step 1 or step 7 ever diverge, even by one bit. That's what makes cross-platform determinism the central engineering problem.

## Why this is hard (cross-machine determinism)

Mac, Linux, x86, and ARM all run "the same" PyTorch code, but underneath:

- Different SIMD widths cause different summation orders inside matmul, so `a + b + c + d` on Mac may be `(a + b) + (c + d)` and on x86 `((a + b) + c) + d`. Floats are not associative — those two expressions can differ in the last bit.
- Different libm vendors round transcendentals (`exp`, `sqrt`) differently in the last 1–2 bits.
- Different thread counts inside BLAS cause different reduction trees.

A single ULP of difference in a logit gets amplified by softmax + arithmetic coding, and the resulting blob is no longer decodable on the other machine. This is the central problem Rung 2 solves.

The fix is to represent everything as 32-bit signed integers (`Q15` scale: each value is `round(x * 2^15)`). Then:

- **Integer addition IS associative.** `(a + b) + c == a + (b + c)` in `int32`/`int64`, every time. SIMD width and thread count stop mattering.
- **Integer matmul** is just integer mul + accumulate. Same answer on every platform.
- **Tricky ops** (softmax, sqrt, exp, layernorm, GELU, Adam's bias correction) all get reimplemented in pure integer math: Taylor series for exp, Newton's method for isqrt, range reduction, careful precision tracking with Q30/Q31/Q46 intermediate scales.

Result: `kolmo` in fixed mode produces a SHA-256-identical blob on Mac, Windows, Linux, x86, and Apple Silicon. The CI workflow proves this on every push.

## Architectural pieces

- **Stable initialization** (`stable_init.py`): a custom SplitMix64 PRNG fills weights so two different PyTorch versions / OSes get the same starting bytes. PyTorch's `manual_seed` is reproducible per-platform but not across platforms; this is.
- **Weight tying**: the `(vocab, d_model)` token embedding and the `(d_model, vocab)` output head share one matrix. Standard modern-LM trick; ~65 K parameters dropped from a ~2 M total, and gradients from either side improve the shared tensor.
- **Sliding-window KV cache** (PyTorch and fixed mode both): per-byte inference cost drops from O(T²) to O(T) where T is the context length. The fixed-mode cache is bit-identical to running `fixed_forward` over the same history — proven by a test that compares warm + step against full forward at the bit level.
- **Rolling-hash copy matcher + adaptive copy models**: the compressor indexes 8-byte keys in a bounded 64 KB window, then encodes `(offset, length)` copy events with adaptive offset / length / event-flag distributions. The decoder only needs to replay the encoded copies; both sides update the adaptive distributions in lockstep.
- **Adaptive byte-context literal model**: literals are encoded under a mixture of transformer probabilities and a mirrored byte model: 50% dense order-2, 3% order-1, 0.5% order-0, remainder neural. The order-2 table is bounded at 64 MB (`65536 * 256 * uint32`) and learns enwik's local markup/text byte transitions immediately, including from copied bytes.
- **RoPE positional encoding**: PyTorch mode defaults to rotary position embeddings instead of learned absolute `pos_emb`; this removed the dead position table and improved enwik prefix ratios.
- **Deterministic-quantized probabilities** (`det_probs.py`): logits are snapped to a 1/64 grid and converted to integer frequencies on a `2^16` total before they touch the arithmetic coder. This isolates the coder from any residual float drift in PyTorch mode.

## Speed (May 2026)

Numbers from the Mac dev machine. ~3.4 M param model (d_model=256, n_heads=8, n_layers=4, max_context=512, tied head/embedding):

| Phase | PyTorch | Fixed |
|---|---|---|
| Inner-loop compress per byte | ~14 ms | ~95 ms |
| Inner-loop decompress per byte | ~9 ms | ~95 ms |
| Seed warmup (cache miss, full corpus) | ~5 s | ~45 s |
| Seed warmup (cache miss, tiny test corpus) | <1 s | ~3 s |
| Seed warmup (cache hit) | not cached | ~0.9 s |
| Round-trip on 62-byte payload (skip prime) | ~1 s | ~5.7 s |

The PyTorch inner loop runs a forward+backward+Adam per block; that's already ~50 ms per block dominated by float32 matmul. Fixed mode does the same dance, but matmul is now routed through float64 BLAS (bit-identical to int64 for our value ranges — products fit exactly in float64's 53-bit mantissa, so no rounding losses, so reorderings don't change the result).

Where the time goes now (from cProfile on the 5.7s round-trip):

- `fixed_adam_step`: ~2.9 s — per-tensor Adam moments + bias correction
- `isqrt_vec`: ~1.5 s (Newton sqrt used by LayerNorm and Adam's `sqrt(v)`)
- `matmul` (now float64 BLAS): not in the top profile entries
- Everything else: ~1.3 s

Speed has improved ~6× over the course of Rung 2 work via:
- Float64-BLAS matmul (bit-identical via mantissa-safe accumulation): -7 s
- Fixed-point KV cache (was recomputing the full forward per byte): -10 s
- `isqrt_vec` bit-length seed + ufunc dispatch for small arrays: -13 s
- `max_context` 16384 → 512 (pos_emb was 99% dead weight in Adam): -5 s
- Lazy block training (skip the final train step when no future byte can benefit): -3 s
- Pre-computed seed cache: 3 min → 1 s startup amortized

Compression ratio on a 246-byte English snippet, both modes without seed prime (a deliberately unfair regime that exposes how each mode handles random-init):

| Mode | Output / Input | Notes |
|---|---|---|
| PyTorch (no prime) | 432 / 246 = 1.76 | random init produces *confident* wrong logits; arithmetic coding pays many bits per miss |
| Fixed Q15 (no prime) | 184 / 246 = 0.75 | Q15 quantization clamps the extreme logits, so random-init failures cost less |

With prime, the PyTorch mode pulls ahead on accuracy and the gap closes; the fixed mode pays a tiny ratio cost (< 1 pp on most regimes) for its determinism guarantee.

## Compression ratio (PyTorch path, enwik9 prefixes)

Measured on ElliePC (RTX 4060 Ti), full seed warmup, RoPE, cost-aware copy selection, and the tuned order-2 byte-context literal model. These are still tiny compared with enwik9's full 1 GB, but they are real enwik bytes and the curve improves with size.

| Prefix | gzip -9 | kolmo | Delta |
|---:|---:|---:|---:|
| 16 KB | 6,266 B / 3.060 bpb | **5,980 B / 2.920 bpb** | -4.6% |
| 32 KB | 12,501 B / 3.052 bpb | **11,836 B / 2.890 bpb** | -5.3% |
| 64 KB | 24,623 B / 3.006 bpb | **23,272 B / 2.841 bpb** | -5.5% |
| 128 KB | 46,944 B / 2.865 bpb | **44,596 B / 2.722 bpb** | -5.0% |

The current bottleneck is speed, not whether the ratio direction works: the 128 KB no-decode run took ~23.5 minutes on the RTX 4060 Ti path. Full enwik9 needs more kernel work and/or a less expensive training schedule before it is practical.

## Development

```sh
git clone https://github.com/EwoudVV/nnzip-kolmo
cd nnzip-kolmo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# Fast tests (~5s)
pytest tests/ --ignore=tests/test_roundtrip.py

# Full suite including slow integration round-trips (~3 min)
pytest

# Just the cross-machine determinism probes
python benchmarks/determinism/hash_fixed_compress.py
```

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `KOLMO_FIXED` | `0` | `1` routes through the bit-deterministic Q15 integer engine |
| `KOLMO_SKIP_PRIME` | `0` | `1` skips the seed warmup (random-init, useful for fast tests) |
| `KOLMO_NO_SEED_CACHE` | `0` | `1` forces fixed-mode prime to re-run, ignoring `~/.cache/kolmo/` |
| `KOLMO_CACHE_DIR` | `~/.cache/kolmo` | override the primed-state cache location |
| `KOLMO_DEVICE` | auto | force PyTorch path to `cpu` or `cuda` |

## License

MIT
