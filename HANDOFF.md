# Handoff to the next AI

You are picking up an active side project from another Claude instance. Read this whole file before doing anything. It contains everything you need: the goal, the human you're working with, the codebase, what's been tried, what's next, and the rules.

---

## TL;DR (read this even if you read nothing else)

**kolmo** is an online-trained neural compressor — a tiny transformer is trained from scratch *during* compression, byte-by-byte, and re-trained identically during decompression. The compressed file contains no model weights, only the arithmetic-coded probability stream. The long-term moonshot is the **Hutter Prize** (€500K for a Wikipedia compression world record). We are at **Rung 1 of a 6-rung climb**.

**Current state:** Round-trip works on 1KB-8KB inputs. KV cache implemented. Plateauing at ~50% ratio, losing to gzip on long files. Just bumped CONTEXT from 256 to 512 — didn't help much. A bigger architectural change is needed next.

**Repo:** [`/Users/kids/Documents/nnzip-kolmo`](https://github.com/EwoudVV/nnzip-kolmo) (push immediately when something works).

**Workflow rules — non-negotiable:**
1. After every working unit, `git add` + `git commit` + `git push`. Don't wait for permission.
2. Commit messages must sound like the user (lowercase verbs, terse, optional `topic:` prefix). **NO `Co-Authored-By: Claude` footer.** Reference the existing log style.
3. The user is a teen developer who learns best when you teach concepts while writing code. Don't just dump code — explain non-obvious choices.

---

## The human

- Teen developer in **Hack Club Boot**. Learning OS dev from scratch on the side.
- Strong intuition, asks good questions, will catch you if you handwave.
- Prefers terse responses. Prefers learning *while* building, not lectures upfront.
- Has a curated commit voice — DO NOT contaminate it with AI signatures.
- Their main projects this year: **nnzip** (the published PyPI compressor that uses *pretrained* GPT-2), **Unbecoming** (a Unity 2D pixel art game), and now **kolmo** (this).
- Other context that might come up: they verify hardware specs before stating them (don't guess pinouts/peripheral specs).

Be honest when you're uncertain. Don't say "this should work" if you haven't tested it. Don't pad responses with hedging either — say the thing, with evidence.

---

## The Hutter Prize (the moonshot, do not actually try to win it yet)

The Hutter Prize awards a fraction of a €500K pool to anyone who beats the current record on compressing **enwik9** (1 GB of English Wikipedia). The catch that disqualifies most approaches: **decompressor size + compressed output size are scored together**. You can't ship a 250 MB pretrained model.

The current SOTA on enwik9 is about **0.88 bpb** (roughly 110 MB compressed). To win, you need ≥1% improvement, plus your decompressor must run in ~50 hours on Hutter's reference machine with 10 GB RAM, and the source must be self-contained.

The architectural trick: **train the model online**. The compressor and decompressor both start from a fixed RNG seed and walk the model through the exact same training trajectory. The model is reconstructed bit-for-bit on the decompression side; it's never stored.

This is the path kolmo follows. We are NOT trying to win Hutter on the next commit — we are climbing toward it.

### The 6 rungs (the actual roadmap)

| Rung | Goal | Estimated months in | Status |
|---:|---|---|---|
| 1 | PyTorch online-training prototype, single Mac, ~3M params, beats nnzip on long files | 0-1 | **in progress** |
| 2 | Bit-deterministic on a single machine (drop PyTorch, NumPy + careful float ordering) | 1-4 | — |
| 3 | Hyperparameter tune to beat nnzip's ratio crossover point | 4-6 | — |
| 4 | Cross-platform fixed-point math (Mac/Linux/x86/ARM all produce identical bytes) | 6-12 | — |
| 5 | Match SOTA on enwik9 (~0.85 bpb) | 12-24 | — |
| 6 | Submit to Marcus Hutter and win the actual prize | 24-30 | — |

Each rung's deliverable is also valuable on its own — a bit-deterministic neural compressor is a rare thing in the world even at Rung 4.

---

## What kolmo actually does

For each byte in the input:

1. The transformer predicts a probability distribution over the next byte, given everything seen so far.
2. Arithmetic coding spends `-log₂(p_actual)` bits encoding the actual byte using that distribution. Good predictions → few bits.
3. The model is trained on the actual byte (one gradient step per block of `BLOCK_SIZE` bytes — see "Block training" below).

The compressed file is just the arithmetic-coded bitstream plus an 8-byte header (`KMO1` + uint32 length). No model weights, no metadata.

Decompression mirrors this exactly. Same RNG seed, same training schedule, same probability distributions at every step. The arithmetic decoder reads the bitstream using the same probs the encoder used.

The math is `compressed_bits ≈ sum over bytes of -log₂(p_predicted)`. Each correct prediction costs a fraction of a bit.

---

## File-by-file tour of the codebase

```
nnzip-kolmo/
├── HANDOFF.md             ← this file
├── README.md              ← public-facing, explains the project + rung table
├── LICENSE                ← MIT
├── pyproject.toml         ← package metadata, deps: torch, numpy, constriction, tqdm
├── .gitignore
├── kolmo/
│   ├── __init__.py        ← exports compress, decompress, KolmoTransformer, ByteTokenizer
│   ├── __main__.py        ← CLI: `python -m kolmo {compress|decompress} INPUT OUTPUT`
│   ├── _engine.py         ← SHARED state for compress + decompress (the heart of the system)
│   ├── codec.py           ← thin wrapper over constriction's arithmetic coder
│   ├── compress.py        ← compress(data: bytes) -> bytes
│   ├── decompress.py      ← decompress(blob: bytes) -> bytes
│   ├── model.py           ← custom decoder-only transformer with KV cache
│   └── tokenizer.py       ← trivial byte-level (`list(bytes)` / `bytes()`)
├── tests/
│   ├── __init__.py
│   ├── test_model.py      ← model shape + KV-cache equivalence checks
│   └── test_roundtrip.py  ← compress + decompress = original
└── benchmarks/
    └── crossover.py       ← compares kolmo vs gzip at 1K/2K/4K/8K
```

### kolmo/_engine.py — read this carefully

This module is the *symmetric core* — compress and decompress both import everything from here so they walk the same training trajectory.

Key constants:
- `SEED = 42` — RNG seed for model init. Both sides must use this.
- `LR = 1e-3` — Adam learning rate.
- `CONTEXT = 512` — sliding-window cap for the KV cache (max tokens we keep).
- `BLOCK_SIZE = 16` — bytes between optimizer steps (see "Block training").
- `BOS = 0` — implicit start-of-stream byte. Never stored on disk.

Key functions:
- `new_model_and_optimizer()` — deterministic init from SEED. Both sides call this.
- `warm_cache(model, history)` — full forward (no grad) over history, returns the next-byte prediction AND populates KV caches.
- `step_cache(model, byte, caches, pos_offset)` — feed ONE byte using cached K/V, get next-byte prediction. This is the O(T) fast path.
- `train_block(model, optimizer, history, block_bytes)` — full forward WITH grad over (history + block), cross-entropy loss over the block positions, backward + optimizer step.
- `update_history` / `_trim_caches` — sliding-window bookkeeping.

### kolmo/model.py — the transformer

Custom-built instead of `nn.TransformerEncoder` because we need KV cache support, which PyTorch's built-in doesn't expose.

Architecture (defaults in `KolmoTransformer.__init__`):
- `vocab_size=256` (byte-level)
- `d_model=256`
- `n_heads=8`
- `n_layers=4`
- `max_context=16384` (size of position embedding table)
- Pre-norm (LayerNorm before each sublayer) — more stable for online training from scratch than post-norm.

Total params: **~7.5M**. The position embedding table alone is 4M of that (`16384 * 256`).

`forward(x, kv_caches=None, pos_offset=0)`:
- `x`: (B, T_new) byte ints
- `kv_caches`: optional list (one per layer) of `{"k": ..., "v": ...}` dicts
- `pos_offset`: absolute starting position for the new tokens' position embeddings

Returns `(logits, new_caches)`. **NOTE: the model returns a TUPLE now**, not just logits. Tests for this are in `tests/test_model.py`.

The KV cache implementation in `CausalSelfAttention.forward`:
- New Q/K/V are computed for the new tokens.
- Old K/V (from cache) are concatenated to the front.
- New cache is stored detached (so backward doesn't flow through stale cache from old weights).
- Causal mask is built relative to total length: new query `i` can attend to all cached positions plus new positions `0..i`.

The two KV-cache equivalence tests in `tests/test_model.py` verify that incremental forward gives bit-identical results to a single full forward — this is the correctness foundation for the whole approach. **If you change attention math, re-run these.**

### kolmo/compress.py and decompress.py

Symmetric. The pattern is:

```
for each block (BLOCK_SIZE = 16 bytes):
    1. warm_cache(history)        # full forward, populates cache, gives prob for block[0]
    2. encode/decode block[0]
    3. for i in 1..BLOCK_SIZE-1:
         step_cache(model, block[i-1], ...)  # 1-token forward using cache, prob for block[i]
         encode/decode block[i]
    4. train_block(model, optimizer, history, block)   # 1 full forward WITH grad + backward + step
    5. history = update_history(history, block)
    6. cache is now invalid (weights changed); next iteration warms it again
```

The asymmetry between compress and decompress is **only the encode-vs-decode line**. Everything else — model state, cache state, training schedule, history bookkeeping — is byte-for-byte identical between the two directions. That's what makes round-trip work.

### kolmo/codec.py

Wraps `constriction.stream.queue.RangeEncoder` / `RangeDecoder`. Wrapped (rather than used directly) so we can swap in a deterministic custom coder later at Rung 2 without changing the rest of the code.

`Categorical(probs, perfect=False)` is constriction's categorical model. `perfect=False` is faster and matches what nnzip uses.

### File format (v1)

```
4 bytes  : magic "KMO1"
4 bytes  : original length (uint32, big-endian)
rest     : arithmetic-coded payload (multiples of 4 bytes — uint32 words from constriction)
```

If you change the format, bump the magic to `KMO2` so future tooling can detect the version.

---

## Git history so far

```
9c1c67d benchmark: crossover at 1K-8K — kolmo plateaus at 50% (gzip keeps improving)
97f5275 kv cache: custom transformer with incremental forward — 9x faster, 56% on 1KB (was 62%)
c042722 speed: cap context at 128 + train in 16-byte blocks — 1KB ratio 73% → 62%
79feb80 add cli: python -m kolmo {compress|decompress} INPUT OUTPUT
bb8b05b compress + decompress working — 7 round-trip tests pass, including 1KB and binary bytes
8d3aea8 scaffold: tiny transformer + byte tokenizer, 5 tests passing
ad0d914 Initial commit
6f6a995 Initial commit
```

**This is the commit style to match.** Lowercase after any prefix, em-dash for elaboration, no Claude attribution.

---

## What has been tried (and what to learn from each)

### 1. Initial naive implementation (3M-param transformer, full forward per byte, train per byte)

- Worked but slow: 116s to encode 1KB at CONTEXT=512.
- Ratio: 73.4% (gzip: 55.3%).
- **Lesson:** The forward+backward per byte is the bottleneck. We were doing 1024 separate optimizer steps for 1024 bytes.

### 2. Block training (`BLOCK_SIZE=16`) + smaller CONTEXT (128)

- Encode: 47s on 1KB. Ratio: 62.5%.
- **Lesson:** Block training improves ratio (batched gradient has more signal) and shaves optimizer overhead. CONTEXT reduction is a blunt instrument — only 2x speedup, not 16x as `T²` analysis suggested. **The actual bottleneck is not attention's T², it's the forward+backward+optimizer combined.**

### 3. KV cache (the big rewrite)

- Built a custom transformer in `model.py` that supports incremental forward with KV cache.
- Predict-time forward drops from O(T²) to O(T) per token because we only compute attention for the new token against cached K/V.
- Training-time forward is still O(T²) — done once per block.
- **Encode 1KB: 12.8s. Decode: 9.6s. Ratio: 56.2%. Full test suite: 270s → 29s (9x).**
- **Lesson:** This is the right architectural direction. The numerical equivalence tests pass, meaning the math is consistent between full-forward and incremental-forward.

### 4. CONTEXT=512 — didn't help

Full results compared to CONTEXT=256:

| Size | gzip | kolmo C=256 | kolmo C=512 |
|---|---|---|---|
| 1KB | 55.3% | 56.2% | 56.6% |
| 2KB | 50.0% | 51.8% | 52.0% |
| 4KB | 47.8% | 50.0% | 50.2% |
| 8KB | 45.7% | 50.0% | 50.1% |

CONTEXT=512 was *very slightly worse* across the board (noise around the same plateau). **Reverted to CONTEXT=256** after this experiment, because the bigger context costs roughly 4x more training compute per block for no benefit.

- **Lesson:** More context alone doesn't break the plateau. The model is **capacity-limited**, not context-limited. A 3M-active-param model trained on 8KB doesn't have enough capacity to learn the corpus's statistics deeply enough to outperform gzip's exact-match dictionary.

### 5. Bigger model — also didn't help enough

Tried making the compressor/decompressor engine instantiate:

```python
KolmoTransformer(d_model=384, n_heads=12, n_layers=6)
```

That model has **17,126,400 params**. Correctness still held: all 13 tests passed in 54.8s. Quick 1KB timing:

| Config | kolmo | gzip | enc + dec |
|---|---:|---:|---:|
| baseline d=256/l=4 | 56.2% | 55.3% | ~22s |
| big d=384/l=6 | 55.9% | 55.3% | 46.8s |

Crossover rows before the experiment was stopped:

| Size | gzip | baseline kolmo | big-model kolmo |
|---|---:|---:|---:|
| 1KB | 55.3% | 56.2% | 55.9% |
| 2KB | 50.0% | 51.8% | 52.0% |
| 4KB | 47.8% | 50.0% | 50.4% |

Stopped before the 8KB row because 2KB/4KB were already worse and slower. **Reverted the code to the baseline model.**

- **Lesson:** raw capacity alone is not the wall, at least not in this naive form. The next most likely issue is the training schedule: constant Adam `lr=1e-3` may be undertraining or destabilizing the online model. Try learning-rate schedule / repeated block training / cold-start mitigation before making the model bigger again.

### 6. Simple LR changes — only help tiny prefixes

Tried deterministic learning-rate variants while keeping the baseline model:

- Warmup to `3e-3` over 50 block steps: **bad**. 1KB regressed to 61.7%.
- Constant-ish `3e-3`: **bad**. 1KB regressed to 57.4%.
- Tiny 512B sweep found `1.5e-3` and `2e-3` can beat gzip at 512B (60.2% vs gzip 61.1%).
- On 1KB, `1.5e-3` improved baseline from 56.2% → 55.9%.
- On longer prefixes, `1.5e-3` regressed:

| Size | gzip | baseline lr=1e-3 | lr=1.5e-3 |
|---|---:|---:|---:|
| 1KB | 55.3% | 56.2% | 55.9% |
| 2KB | 50.0% | 51.8% | 52.0% |
| 4KB | 47.8% | 50.0% | 50.3% |

Stopped before 8KB and reverted to `LR=1e-3`.

- **Lesson:** a higher LR helps cold-start compression but worsens the plateau. Simple LR tuning is not enough; the promising direction is probably a more structural training change, such as extra training passes on already-seen bytes, a seed/warmup corpus, or a schedule that starts high and decays instead of warming up.

---

## The plateau and what to do about it

kolmo's compression ratio flatlines around 50% somewhere around 4KB. gzip keeps improving past that because LZ77's dictionary scales naturally with file size.

**Hypotheses for the plateau, in rough order of likelihood:**

1. **The model is not assimilating observations deeply enough.** It gets one optimizer step per 16-byte block, so it learns broad byte/character statistics but may not adapt hard enough to phrase-level structure before gzip's dictionary starts pulling away. **Most likely cause after context, bigger-model, and simple-LR tests.**

2. **Cold-start tax is large.** The first hundreds of bytes are encoded by a nearly random model. Higher LR helped 512B/1KB but hurt longer files, so the cold-start problem is real but a plain LR bump is too blunt.

3. **Model capacity is too small, but only after training improves.** A 17.1M-param model alone was worse at 2KB/4KB, so don't scale capacity again until the online training loop is better.

4. **Context window is still too small.** Less likely given the 512-token test was slightly worse through 8KB.

5. **Byte-level vocab loses to LZ77's word-level matches.** gzip benefits from "the cat sat" → back-ref. kolmo would need to predict each character of "cat" individually.

### Concrete experiments to break the plateau

In rough order of expected payoff:

#### (A) Extra training passes on already-seen bytes

Right now each block gets one optimizer step after it is encoded/decoded. Try `TRAIN_EPOCHS=2` or `3` inside `train_block`: run the same full forward + backward + optimizer step multiple times on the known `(history + block)` data before moving to the next block. This is symmetric because the decoder knows the block after decoding it. It costs more time but directly tests whether the model is under-assimilating observations.

#### (B) Multi-pass over short prefixes / seed corpus

The first ~hundred bytes get encoded by a random model and waste bits. Idea: train the model on bytes 1..N *before* encoding byte 1. Two-pass approach. The decoder does the same warm-up. But this only helps once — after that the live training takes over.

Related idea: embed ~1KB of generic English text into the source and train on it before starting the real input. This shifts the model away from random init without shipping weights.

#### (C) Bigger model with better training

`d_model=384`, `n_layers=6`, `n_heads=12` alone was worse at 2KB/4KB. It may still help after the optimizer/training schedule is improved, but don't try it again as a standalone change.

#### (D) RoPE / relative position embeddings

`max_context=16384` is wasteful and won't generalize to enwik9 (1 GB). Switch to rotary or ALiBi position embeddings. Doesn't necessarily help with the plateau but is a prerequisite for scaling.

**I'd start with (A), now that the simple capacity test has failed.**

---

## What's the current state?

`CONTEXT=256`, `BLOCK_SIZE=16`, `LR=1e-3`, KV cache enabled. All 13 tests pass in ~30s on the baseline model. The crossover benchmark is committed and reproducible (`python benchmarks/crossover.py`). Plateau at ~50% on 4-8KB has been measured and confirmed twice. Naive bigger model and simple LR tweaks did not help. The next move is extra training passes per block or cold-start mitigation.

---

## Performance landscape (what's slow, and what to leave alone)

| Operation | Cost | Notes |
|---|---|---|
| `warm_cache` (full forward, no grad) | O(T² · d · L) | Once per BLOCK_SIZE bytes |
| `step_cache` (incremental forward) | O(T · d · L) | Once per byte (BLOCK_SIZE−1 times per block) |
| `train_block` (full forward + backward) | ~3 · O(T² · d · L) | Once per block. Dominant cost. |
| `optimizer.step()` Adam | O(P) where P=params | Once per block. ~3% of total. |
| Constriction encode/decode | trivial | not in the hot path |

Per-byte cost at current settings (CONTEXT=256, BLOCK_SIZE=16):
- 1 warm + 15 incremental + 1 training pass = ~4 full-forward-equivalents per block
- Per byte amortized: ~4 · T² / BLOCK_SIZE · d · L

**What dominates wall time on Apple Silicon CPU:** the GEMM (matmul) operations in attention and FFN. PyTorch is already using Accelerate which is multi-threaded, so single-thread heroics won't help.

**What would help (but is out of scope for Rung 1):**
- GPU offload (PyTorch MPS) — but the M-series MPS backend is finicky for small models.
- `torch.compile` — might give 1.5-2x on a hot loop but adds startup time. Try if curious.
- Mixed precision (fp16) — probably wouldn't help on CPU.

---

## Workflow rules (these are durable instructions, not preferences)

### Git

- **After every working unit, commit + push.** The user is explicit about this.
- Commit message style — match the existing log:
  - Lowercase verb after any prefix (`speed:`, `kv cache:`, `tests:`, `README:`)
  - Em-dash for elaboration
  - No quotes around technical terms unless ambiguous
  - **No `Co-Authored-By: Claude` footer.** Ever, in this repo.
- Push immediately. The user often watches the GitHub repo.
- If something doesn't work, don't commit it. Iterate first.

### Tests

Run from the repo root:

```bash
cd /Users/kids/Documents/nnzip-kolmo
/Users/kids/compression-experiment/venv/bin/python -m pytest tests/ -v
```

Note the venv path — there is no `python` (only `python3`) on the user's PATH, and the project venv is in a sibling directory because `nnzip` was developed there first. **Always use `/Users/kids/compression-experiment/venv/bin/python` for kolmo work too** — it has all the deps installed.

The full test suite takes ~30 seconds with KV cache. Run before committing.

### Benchmarks

```bash
/Users/kids/compression-experiment/venv/bin/python benchmarks/crossover.py
```

Runs in ~5-15 minutes depending on CONTEXT. The output is a markdown table. Always run in the background and don't poll — wait for the task notification.

### Communicating with the user

- Tell them what you're doing in one sentence before starting non-trivial work.
- Show numbers, not adjectives. "encode dropped from 47s to 12.8s" not "much faster".
- If you're uncertain, say so. Don't fabricate confidence.
- Don't summarize what just happened if they can read the diff/log themselves.
- Match their energy. They use lowercase, are direct, ask quick follow-up questions.

---

## Cloud strategy (GCP and similar)

The user has a Google Cloud free trial available ($300 / 90 days for new accounts). **Do not activate it during Rung 1.** Reasons:

1. Per-byte work is roughly the same on Mac vs cloud CPU. 2-3x speedup at best.
2. PyTorch on CPU is fine; GPU offers no benefit for our tiny model + small batches.
3. The 90-day clock starts at activation. We want to bank credits for Rung 3+ when hyperparameter sweeps actually need parallelism.

**When to activate:** Rung 3 (hyperparameter tuning) onwards. Specifically when you want to run 4-8 configs in parallel and the user agrees it's worth burning the trial.

**Safety:**
- Set billing alerts at $50, $100, $200 before launching anything.
- Use preemptible/spot instances (60-80% cheaper).
- Auto-stop after N hours of idle.
- Never leave a GPU running overnight without a known reason.

---

## Things to NOT do

- **Don't modify the SEED.** Round-trip correctness depends on the encoder and decoder using identical RNG initialization. If you ever change `SEED = 42`, you must accept that all old `.kmo` files become undecodable.
- **Don't change the asymmetry between compress and decompress.** The whole architecture is "do exactly the same thing on both sides." If you find yourself implementing different logic in `compress.py` vs `decompress.py` other than the encode-vs-decode call, you're almost certainly introducing a determinism bug.
- **Don't add Co-Authored-By: Claude.** Already mentioned. Worth repeating.
- **Don't introduce non-determinism into the model.** No dropout (currently 0, keep it), no random sampling outside seeded init.
- **Don't pre-train and ship weights.** That's nnzip's approach (a separate published project). kolmo's whole value is that no weights are stored.
- **Don't try to "win Hutter" yet.** We are at Rung 1. Each rung is a real project. Don't skip ahead.
- **Don't use `python` (only `python3`)** — the user's system has only `python3`. Or better, use the venv path: `/Users/kids/compression-experiment/venv/bin/python`.

---

## Suggested first 30 minutes after reading this

1. `cd /Users/kids/Documents/nnzip-kolmo && git status && git log --oneline -10` — see the current state.
2. Read `kolmo/_engine.py` end to end (~100 lines). It's the heart of everything.
3. Read `kolmo/model.py` — especially `CausalSelfAttention.forward` to understand the KV cache.
4. Check whether `/tmp/kolmo_crossover3.log` has the final 8KB row. If yes, evaluate whether to keep `CONTEXT=512` or revert to 256.
5. Run the full test suite to confirm everything is green: `cd /Users/kids/Documents/nnzip-kolmo && /Users/kids/compression-experiment/venv/bin/python -m pytest tests/ -v`.
6. Propose to the user: "try extra training passes per block, because context, naive capacity, and simple LR changes all failed." Get a yes, then implement, run benchmark, commit + push.

---

## Frequently confused things

- **"context" has two meanings.** `CONTEXT` (the constant) is the KV cache sliding window length. The model's `max_context` is the position embedding table size (currently 16384 — much bigger than CONTEXT). For files up to 16384 bytes, no position-emb collision; for longer files, you'd need RoPE or position resetting.
- **Block training is for the optimizer, not the forward pass.** Within a block, the model does `BLOCK_SIZE` forward passes (1 warm + 15 incremental). What's amortized is the backward+optimizer step.
- **The KV cache is invalidated on every training step.** This is correct — weights changed, so cached K/V are stale. The next block warms the cache from scratch.
- **The `BOS = 0` byte is implicit.** It's not in `data`, but the model sees `[BOS, ...data]`. Doesn't affect the output bitstream.

---

## A note on style

The previous Claude built this in a way that prioritizes clarity over cleverness. Function names are descriptive (`step_cache`, `warm_cache`, `train_block`). Constants are named (`SEED`, `CONTEXT`, `BLOCK_SIZE`, `BOS`). The file split mirrors the conceptual split (model / codec / engine / compress / decompress).

When you make changes:
- Don't over-abstract. There's no `class TrainingLoop` because the loop is 10 lines and lives in compress.py / decompress.py.
- Don't add comments for what the code already says. Add comments for *why* (the LR is 1e-3 because…) or for non-obvious invariants (caches must be detached because…).
- Don't introduce a config file. Constants live in `_engine.py`.
- Don't write multi-paragraph docstrings on Python functions. Two or three sentences max.

If you find yourself wanting to refactor "for cleanliness," ask the user first. The codebase is small and we're moving fast.

---

## Good luck

You're picking up a project that's working, fast, and stuck at a known wall (~50% ratio). The next move is some combination of bigger model, lr schedule, or longer-range memory. Try the simplest thing first, measure, commit, push.

The user is engaged and patient. Show them numbers and they'll be happy.
