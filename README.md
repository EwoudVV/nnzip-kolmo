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
   - Blend that neural distribution with an adaptive byte-context side model (PPM-C escape over order-4/2/1/0 by default).
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
- **Adaptive byte-context literal model**: literals are encoded under a blend of transformer probabilities and a mirrored byte model. The default (`KOLMO_LITERAL=ppm`) is a **PPM-C escape blend**: walk the longest available order (4 → 2 → 1 → 0 with optional 3/5), at each order spend `p(b) = count[b] / (sum + distinct)` for bytes seen there and `escape = distinct / (sum + distinct)` for everything else. Each byte pays the escape cost only for orders that didn't match, instead of paying a static blend weight every time. The final blend weight is **cost-aware** (`KOLMO_ADAPTIVE_WEIGHT=1`, default on): when PPM's distribution is sharply peaked on one byte (max(p_ppm) close to 1) the neural model is mostly noise and we let PPM dominate (weight `LITERAL_NEURAL_WEIGHT_LOW=0.20`); when PPM is near-uniform (cold contexts) the neural model is the only signal and gets `LITERAL_NEURAL_WEIGHT_HIGH=0.70`. On a 16 KB enwik9 prefix the full preset lands at 2.906 bpb (with the default order-3 walk also on) — −0.047 bpb vs the legacy `KOLMO_LITERAL=mix` fixed-weight blend (40% order-2 + 20% hashed order-4 + 2% order-1 + 0.5% order-0 + remainder neural), which is still available as an opt-in. Disable adaptive (`KOLMO_ADAPTIVE_WEIGHT=0`) to fall back to a fixed `KOLMO_NEURAL_WEIGHT=0.50` blend for historical comparison. The dense order-2 table is bounded at 64 MB (`65536 * 256 * uint32`); the order-4 table is a fixed-size hashed table with SplitMix-style bucket mixing so repeated wiki markup/text contexts are learned immediately without unbounded memory.
- **RoPE positional encoding**: PyTorch mode defaults to rotary position embeddings instead of learned absolute `pos_emb`; this removed the dead position table and improved enwik prefix ratios.
- **Deterministic-quantized probabilities** (`det_probs.py`): logits are snapped to a 1/64 grid and converted to integer frequencies on a `2^16` total before they touch the arithmetic coder. This isolates the coder from any residual float drift in PyTorch mode.

## Speed (May 2026)

Numbers from the Mac dev machine. ~3.4 M param model (d_model=256, n_heads=8, n_layers=4, max_context=512, tied head/embedding, RoPE default):

| Phase | PyTorch | Fixed |
|---|---|---|
| Inner-loop compress per byte (4KB skip-prime) | ~10.7 ms | ~71 ms |
| Inner-loop decompress per byte | ~9.2 ms | ~71 ms |
| Seed warmup (cache miss, full corpus) | ~70 s | ~45 s |
| Seed warmup (cache miss, tiny test corpus) | <1 s | ~3 s |
| Seed warmup (cache hit) | not cached | ~0.9 s |
| Round-trip on 62-byte payload (skip prime) | ~1 s | ~5.7 s |

Recent PyTorch-path speedups (all bit-equivalent except the schedule change, which deliberately trades training cadence for speed):
- `torch.no_grad()` → `torch.inference_mode()` in inference
- `_causal_mask` cached per `(T_new, T_total, device)` instead of allocated 15k times per 4KB
- `RotaryPositionalEmbedding.apply` uses `stack(...).flatten(-2)` instead of `empty_like + strided writes`
- Manual attention → `F.scaled_dot_product_attention` (small CPU win, would be much bigger on GPU)
- Training schedule doubles every 2 KB instead of 4 KB (caps the per-byte training cost sooner; ratio neutral)

4KB enwik-style compress (skip-prime), running ledger:
- baseline (pre-sprint): 108 s
- after inference path cleanup: 81.8 s (-24%)
- after schedule doubling=2048: 78.1 s (-28%)
- after `KOLMO_MODEL=draft` preset: 50.9 s (-53%) — only ~+0.4 pp ratio cost

### `KOLMO_MODEL` presets

For hyperparameter sweeps you can switch to the draft preset:

| preset | params | 4 KB total | 16 KB enwik total | 4 KB ratio | 16 KB enwik bpb |
|---|---|---|---|---|---|
| `full` (default) | 3.4 M | 78 s | 174 s | 0.4551 | **2.906** (PPM + adaptive + order-3 default) |
| `draft` | 1.4 M | 51 s | 111 s | 0.4590 | **2.922** (PPM + adaptive + order-3 default) |

Default literal pipeline (PPM-C escape over hashed order-4/3 + dense order-2/1/0 + cost-aware adaptive blend) lands the **full** preset at 2.906 bpb on 16 KB enwik9 — a −0.029 bpb improvement over PPM-only with a static 0.50 neural weight and only orders 4/2/1/0 walked, and −0.047 bpb vs the legacy "mix" default. Both presets round-trip identically. The bpb gap between draft and full is now small enough that draft is a perfectly serviceable sweep target — useful for trying copy / literal / schedule knobs at ~2x cadence, then validating the winning config on `full`. Blobs are not interchangeable across presets (different model architectures), so set `KOLMO_MODEL=draft` on both compress and decompress.

The PyTorch inner loop runs a forward+backward+Adam per block; that's already ~50 ms per block dominated by float32 matmul. Fixed mode does the same dance, but matmul is now routed through float64 BLAS (bit-identical to int64 for our value ranges — products fit exactly in float64's 53-bit mantissa, so no rounding losses, so reorderings don't change the result).

Where the time goes now (from cProfile on the 5.7s round-trip):

- `fixed_adam_step`: ~1.4 s — was 2.9 s before A2 fused the 17-stage per-tensor numpy pipeline into one numba pass (2.1x on the draft tensor zoo, bit-identical to the numpy fallback). Round-trip bench in fixed mode dropped accordingly.
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

Latest current-default run (full seed warmup, RoPE, cost-aware copy selection,
length bucket coding, and the hashed order-4 byte-context literal model). These
are still tiny compared with enwik9's full 1 GB, but they are real enwik bytes
and the curve improves with size.

The current 16-64 KB rows below were remeasured locally on the Mac after ElliePC
dropped off the network. Treat them as current-code fallback numbers until the
same run is repeated on ElliePC. The 128 KB row is the previous local fallback
before the SplitMix bucket-mixer retune and is marked pending because the
current 128 KB rerun is slow (~20+ minutes).

| Prefix | gzip -9 | kolmo | Delta |
|---:|---:|---:|---:|
| 16 KB | 6,247 B / 3.050 bpb | **5,920 B / 2.891 bpb** | -5.2% |
| 32 KB | 12,488 B / 3.049 bpb | **11,668 B / 2.849 bpb** | -6.6% |
| 64 KB | 24,589 B / 3.002 bpb | **22,792 B / 2.782 bpb** | -7.3% |
| 128 KB | 46,884 B / 2.862 bpb | **43,780 B / 2.672 bpb** | -6.6% (pending rerun) |

The current bottleneck is speed, not whether the ratio direction works: the
latest 128 KB no-decode run took ~22 minutes locally, and earlier ElliePC runs
were in the same painful 20-25 minute band. Full enwik9 needs more kernel work
and/or a less expensive training schedule before it is practical.

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
| `KOLMO_MODEL` | `full` | `draft` selects a smaller 1.4 M model for faster iteration (set on both compress + decompress; blobs aren't interchangeable) |
| `KOLMO_USE_ROPE` | `1` | `0` falls back to learned absolute position embeddings |

## License

MIT
