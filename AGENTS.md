# nnzip-kolmo — Project Context

## Goal
Build a custom lossless compression algorithm using a PAQ-style ensemble framework with a transformer mixer and hand-designed predictors to beat Hutter Prize contenders.

## Constraints
- Ratio output must be bit-for-bit identical across commits when default settings unchanged (verified by sha256 probe `13e84c65` on 172 B payload)
- Cross-OS determinism: every float reduction uses `math.fsum` to avoid platform-dependent numpy `.sum()` ULP differences
- All predictors and mixers live in `kolmo/_predictors.py`; `LiteralModel` in `kolmo/_engine.py` is a thin orchestrator
- Environment variables gate all optional features; defaults must preserve existing ratio
- Every commit verifies byte-identical output via the 172 B / sha256 `13e84c65` ratio probe
- Full suite green (183+ tests; mixer/predictor coverage in `tests/test_literal_model.py`, codec symmetry in `tests/test_roundtrip.py`)

## Key Design Decisions
- PAQ-style ensemble (many cheap predictors + learned mixer) rather than a pure cmix clone
- `LinearEnsembleMixer` is opt-in via `KOLMO_MIXER=linear`; default stays `CostAwareAdaptiveMixer` (adaptive blend)
- `Predictor.probs()` returns `np.ndarray | None` — None signals "no opinion" so the mixer drops that predictor for the current byte
- `Mixer.combine()` takes `predictor_outputs: dict[str, ndarray | None]` + `neural_probs` — neural is special-cased
- 5 structural predictors shipped opt-in; all default to inactive
- All extra predictors are ignored by the default `CostAwareAdaptiveMixer` — they only activate with `KOLMO_MIXER=linear`/`logistic`
- `KOLMO_PREDICTORS=name,name,...` registers structural predictors on every fresh LiteralModel (no monkey-patching in benches); registry in `kolmo._predictors.EXTRA_PREDICTORS`

## Learned Mixing: byte-wise attempt failed, bit-tree shipped
**Post-mortem (do not retry byte-wise):** the first logistic mixer mixed
stretched logits per byte value, squashed all 256, and renormalized.
Squash produces 256 *independent* Bernoulli estimates; renormalizing
them destroys concentration — a confident neural spike of p=0.5 came out
at p≈0.02 (+4.5 bits wasted per confident byte). Benched 4.92 bpb vs
2.92 cost_aware at 16 KB. Weight renormalization / init / gradient fixes
were all symptom-patches; the structure was wrong.

**`BitTreeLogisticMixer` (KOLMO_MIXER=logistic) is the real PAQ form:**
byte = 8 binary decisions (MSB first). Each predictor's 256-way
distribution induces an exact binary probability at each of the 255 tree
nodes (upper-half mass / block mass, integer-exact). Stretch → mix with
learned per-(level, bucket) weights → squash → multiply factors down
each leaf's path. Output sums to 1 *by construction* — no renorm.
Silent (None) and uniform predictors are exactly logit 0 = no-op, so
the silent-predictor dilution cannot occur. Trained online (8 bit-level
SGD updates per literal) identically on both sides via the
`train_on_literal` hook in compress.py/decompress.py (literal path only,
never copy events). Sharp-prediction sanity: p=0.5 in → 0.4986 out
(1.004 bits vs 1.000 ideal; table-quantization cost only).
Env knobs: `KOLMO_LOGISTIC_BUCKETS` (1/4/16), `KOLMO_LOGISTIC_LR`,
warm start `neural=1.0, ppm=0.4` (override via KOLMO_LINEAR_WEIGHTS).

### Bit-tree bench (draft model, enwik9, KOLMO_SKIP_PRIME=1)
| Config | 8 KB bpb | 16 KB bpb |
|---|---|---|
| cost_aware (default) | 2.8633 | 2.9219 |
| logistic, no extras, b=16 | 2.8008 | — |
| logistic+4p, b=16, lr=.01 | 2.7812 | 2.8496 |
| logistic+4p, b=4, lr=.01 | 2.7422 | — |
| **logistic+4p, b=1, lr=.01** | **2.6797** | **2.7949** |
| logistic+4p, b=1, lr=.02 | 2.6719 | — |
| logistic+4p, b=16, lr=.002 | 2.8633 | — |

Findings: (1) learned mixing alone beats the hand-tuned default by
-0.06 bpb; (2) the 4 structural predictors add real signal on top;
(3) **1 bucket >> 16 buckets at these sizes** — 16 buckets fragments
the training signal (~512 samples per weight set at 8 KB), 1 bucket
converges 16x faster. Re-test bucketed weights at 256 KB+ before
assuming 1 bucket stays optimal. (4) lr=0.002 is too slow to move off
the warm start; lr=0.01 is the working default.

## Bench Results (draft model, PyTorch path, 4 KB enwik9)

| Config | bpb | Bytes | vs Baseline |
|---|---|---|---|
| Baseline (linear 50/50) | 2.0312 | 1040 | — |
| **+balanced_delimiter** | **2.0156** | **1032** | **-8 B** |
| **+after_number** | **2.0156** | **1032** | **-8 B** |
| **+in_text** | **2.0156** | **1032** | **-8 B** |
| **+position_modulo** | **2.0156** | **1032** | **-8 B** |
| +word_fragment | 2.0312 | 1040 | ±0 B |
| +all5_structural | 2.0156 | 1032 | -8 B |

Compound (orthogonal signal separation):
- 1 predictor w=0.03: -8 B (-0.0156 bpb)
- 2 orthogonal (bd+an) w=0.02+0.02: -12 B (-0.0234 bpb)
- 3 orthogonal (bd+an+in) w=0.033 each: -16 B (-0.0312 bpb)
- 4 orthogonal w=0.10 total: -16 B (saturated at 4 KB)

Size scaling (8 KB enwik9, 4 predictors):
- Baseline: 2.8555 bpb (2924 B)
- w=0.04 total: 2.8398 bpb (2908 B) = -16 B
- w=0.10 total: 2.8242 bpb (2892 B) = -32 B

Ratio loss scales linearly with weight and payload size. No regressions.

## Shipped Predictors (all in `kolmo/_predictors.py`)
- **PostCopyPredictor**: copy-match context from LZ77
- **PPMPredictor**: PPM-C escape over orders 5..0
- **WordFragmentPredictor**: word-internal byte transitions via SplitMix64 hash (confirmed negative, reference only)
- **BalancedDelimiterPredictor**: nesting depth of `{}`, `[]`, `()`, `<>` — **+0.77% at 4 KB**
- **AfterNumberPredictor**: 3-state machine (NORMAL/IN_NUMBER/AFTER_NUMBER) — **+0.77% at 4 KB**
- **InTextPredictor**: 2-state XML text vs markup discriminator — **+0.77% at 4 KB**
- **PositionModuloPredictor**: 12-bucket position-since-`\n` state machine — **+0.77% at 4 KB**

## Bench Environment
- Python 3.11 venv at `.venv311/` (numpy<2 for torch compatibility)
- Model presets: draft (1.4M), full (3.4M, default), large (~11M), xl (~26M)
- `KOLMO_SKIP_PRIME=1` for standalone predictor benches
- `KOLMO_MODEL=draft` for fast iteration

## Next Steps
1. More predictors: the mixer auto-weights anything registered, so each
   new predictor idea is now cheap to evaluate (write class, add to
   EXTRA_PREDICTORS registry, bench with KOLMO_PREDICTORS)
2. Mixer context upgrade: feed the partial bit prefix into weight-set
   selection (PAQ-style) once payloads are big enough for bucketed
   weights to win
3. 64-256 KB bench to find where b=16 overtakes b=1
4. Default-flip decision: logistic+4p b=1 beats cost_aware by -0.13 bpb
   at 16 KB; needs a larger-scale bench + full-preset confirmation
   before changing the default (which would change the sha256 probe)
5. Full-model fixed-point bench at scale (requires numba/GPU — blocked)
