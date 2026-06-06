"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. The KV cache lets each direction do most of the predictions
incrementally (O(T) per byte instead of O(T²)); the training step still needs
a full forward over the recent history with gradient tracking, but that's
only done once per block.
"""

import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kolmo.det_probs import TOTAL_FREQ, logits_to_int_freqs
from kolmo.fixed import dequantize
from kolmo.fixed_kv_cache import fixed_step, fixed_warm, trim_caches
from kolmo.fixed_model import extract_fixed_weights, tied_param_pairs
from kolmo.fixed_optim import FixedAdamState
from kolmo.fixed_train import fixed_train_block
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model

SEED = 42
LR = 1e-3
# Linear warmup ramps LR from 0 to LR over the first N optimizer steps.
# Bigger models need this — without warmup, Adam's first few steps with
# zero-initialized m/v moments cause large updates that can destabilize
# training (a known problem at d_model >= ~384). 100 steps ≈ the first
# 1600 bytes of input, well before any training-block-size doubling.
LR_WARMUP_STEPS = 100
_CONTEXT_ENV = os.environ.get("KOLMO_CONTEXT")
CONTEXT = int(_CONTEXT_ENV) if _CONTEXT_ENV else 256  # sliding-window cap (max tokens kept in KV cache)
BLOCK_SIZE = 16  # base bytes between optimizer steps (early in file)
# Sublinear training schedule: training interval doubles every N bytes of
# input seen, capped by CONTEXT-1 so every training slice still has at least
# one preceding token to predict the first byte of the block. Rationale: the
# model adapts fastest in the first few KB. After that each additional Adam
# step contributes less per byte.
#
# Schedule sweep at 4KB (skip-prime, mixed prose):
#   doubling=4096 (old): 4096->1868  ratio=0.4561  total=107.5s
#   doubling=2048      : 4096->1864  ratio=0.4551  total= 92.5s   (-14%, -0.10pp)
# 2048 is a clear win — slightly better ratio AND faster, because halving the
# doubling distance gets the schedule to bigger block sizes sooner, which
# saves more backward calls without losing meaningful early-file adaptation.
_TRAIN_SCHEDULE_DOUBLING_BYTES = 2048
_TRAIN_SCHEDULE_MAX_MULT = 32
# Coalesce multiplier on the schedule threshold. Larger = fewer training
# steps, each one over a proportionally larger block. Effect saturates at
# the CONTEXT-1 cap (when COALESCE * BLOCK_SIZE * mult >= CONTEXT-1 the
# multiplier stops mattering). cProfile on a 2 KB compress showed
# `train_block` is ~57% of total runtime, but the trade between bpb cost
# and wall-time savings turned out unfavorable at small scales:
#
#   16 KB enwik9, skip-prime:
#       coalesce  draft bpb  full bpb  draft time  full time
#       1         2.922      2.906     54.5 s       81.8 s   (baseline)
#       2         2.945      2.934     61.8 s       67.3 s   +0.027 bpb / -18% (full)
#       4         3.016      3.010     39.8 s       60.1 s   +0.10  bpb / -27%
#       8         3.078      3.049     38.9 s       56.3 s   +0.15  bpb / -31%
#
# Even COALESCE=2 erases more bpb than the cost-aware adaptive blend gained
# (-0.012). The ratio cost is the model adapting more slowly — early-file
# updates are where the gradient signal matters most. Default stays at 1.
# Worth re-benching at 1 MB+ scales where the slower adaptation matters
# less and the speed savings are larger in absolute time.
# Both compress and decompress read this value at module load, so they
# stay in lockstep without exchanging state.
_TRAIN_COALESCE = max(1, int(os.environ.get("KOLMO_TRAIN_COALESCE", "1")))


def training_block_size_at(bytes_observed: int) -> int:
    """How many bytes to accumulate before the next optimizer step, given
    that `bytes_observed` bytes have already been processed.

    Both compress and decompress call this with the same argument at the
    same point in the trajectory, so they agree on every training step
    boundary without exchanging any extra state.
    """
    bucket = bytes_observed // _TRAIN_SCHEDULE_DOUBLING_BYTES
    mult = min(1 << bucket, _TRAIN_SCHEDULE_MAX_MULT)
    return min(_TRAIN_COALESCE * BLOCK_SIZE * mult, CONTEXT - 1)
BOS = 0  # implicit start-of-stream byte, never written to disk
COPY_PROB = 0.005
COPY_WINDOW = 65536
COPY_MIN = 8
# COPY_MAX=256 was capping ~75% of copy bytes at the ceiling on 16KB English
# (27 of 29 saturated copies in the structural-repetition regime). Bumping to
# 1024 lets long Wikipedia-style template / citation / header blocks collapse
# into a single copy event instead of N adjacent 256-length copies, each
# paying its own event flag + offset + length header.
COPY_MAX = 1024
COPY_CANDIDATES = 64
# Encoder-side heuristic for copy selection. A copy event is used only if its
# adaptive event+offset+length header costs less than spelling the same bytes
# as literals at this proxy bpb. This is deliberately conservative for enwik:
# current RoPE runs are ~3.1 bpb at 32KB, and long-file literals should get
# cheaper as the model adapts, so short/far copies need to clear a real bar.
COPY_LITERAL_BPB = 2.75
COPY_USE_LITERAL_MODEL_PROXY = False
# Adaptive literal side model mixed into neural byte probabilities. This is
# mirrored by the decoder and costs zero blob bytes. It learns file-local byte
# statistics much faster than the transformer's gradient updates, especially
# for wiki markup and punctuation. Strong mixes hurt enwik; the default is a
# order-2 carries the file-local byte structure; keep small order-1/order-0
# backoff nudges for contexts that are still cold.
# KOLMO_LITERAL chooses the literal-model strategy at module load:
#   "ppm" — PPM-C style escape blend (longest available context, then back off);
#           default. Each byte pays its actual escape cost only for orders that
#           didn't match. -0.018 bpb vs "mix" on 16 KB enwik9 prefix at
#           neural_w=0.50 (full preset: 6048 -> 6012 bytes).
#   "mix" — fixed-weight blend of order-0/1/2/4 with confidence ramps (legacy)
_LITERAL_STRATEGY = os.environ.get("KOLMO_LITERAL", "ppm").lower()
if _LITERAL_STRATEGY not in {"mix", "ppm"}:
    raise ValueError(
        f"KOLMO_LITERAL must be 'mix' or 'ppm', got {_LITERAL_STRATEGY!r}"
    )
# For PPM mode: how much weight to put on the neural distribution when blending
# with the PPM byte-context distribution. 0 = pure PPM; 1 = pure neural.
# 0.50 wins on 16 KB enwik9 for both draft and full presets; re-sweep at larger
# scales as the neural model accumulates more training signal.
LITERAL_NEURAL_WEIGHT = float(os.environ.get("KOLMO_NEURAL_WEIGHT", "0.50"))

# Cost-aware adaptive blend (default on, disable via KOLMO_ADAPTIVE_WEIGHT=0).
#
# Motivation: the static `LITERAL_NEURAL_WEIGHT` ignores how *confident* PPM
# actually is at this byte. When PPM has seen the (prev2, prev) context many
# times and its distribution is sharply peaked on one byte, the neural model
# is mostly adding noise — we should let PPM dominate. When PPM has no useful
# context (e.g., file just started, cold buckets), it falls back to
# near-uniform and the neural model is the only thing carrying signal.
#
# Implementation: read max(p_ppm) as a confidence proxy, then linearly
# interpolate the neural weight between two endpoints:
#   peak ≈ 1/256 (uniform PPM)   → neural weight = LITERAL_NEURAL_WEIGHT_HIGH
#   peak ≈ 1.0   (sharp PPM)     → neural weight = LITERAL_NEURAL_WEIGHT_LOW
# Both bounds independently tunable; the legacy `KOLMO_NEURAL_WEIGHT` knob
# still governs the fixed-weight path so old benchmarks remain comparable.
#
# Validated on 16 KB enwik9 prefix (skip-prime, draft and full presets):
#   draft, fixed w=0.50:   6044 B  bpb=2.9512  baseline
#   draft, adaptive:       6020 B  bpb=2.9395  -0.012 bpb
#   full,  fixed w=0.50:   6012 B  bpb=2.9355  baseline
#   full,  adaptive:       5988 B  bpb=2.9238  -0.012 bpb
# Effect is consistent across LOW in [0.20, 0.30] and HIGH in [0.70, 0.80],
# so the default is at the centre of a flat region — robust, not knife-edge.
_ADAPTIVE_WEIGHT = os.environ.get("KOLMO_ADAPTIVE_WEIGHT", "1").lower() not in (
    "0", "false", ""
)
LITERAL_NEURAL_WEIGHT_LOW = float(
    os.environ.get("KOLMO_NEURAL_WEIGHT_LOW", "0.20")
)
LITERAL_NEURAL_WEIGHT_HIGH = float(
    os.environ.get("KOLMO_NEURAL_WEIGHT_HIGH", "0.70")
)

LITERAL_ORDER2_WEIGHT = 0.40
LITERAL_ORDER1_WEIGHT = 0.02
LITERAL_ORDER0_WEIGHT = 0.005
# 0 means "use the full order-2 weight after the context has been seen once".
# Positive values ramp order-2 trust as count/(count + confidence), useful if
# one-observation contexts overfit.
LITERAL_ORDER2_CONFIDENCE = 2.0
LITERAL_ORDER3_WEIGHT = 0.0
LITERAL_ORDER3_CONFIDENCE = 2.0
LITERAL_ORDER3_BUCKETS = 1 << 16
# Whether the PPM walk (KOLMO_LITERAL=ppm) walks the hashed order-3 context
# table. The legacy `LITERAL_ORDER3_WEIGHT` is a mix-path knob that doesn't
# apply to PPM (PPM uses raw count ratios, not blend weights), so the
# `weight > 0` allocation gate is the wrong gate for PPM — you'd have to
# set a positive mix weight you don't actually want just to get count3 into
# memory. KOLMO_PPM_ORDER3 controls it cleanly.
#
# Default ON: validated at 16 KB enwik9 (skip-prime) it produces a clean
# -0.018 bpb improvement on both draft and full, with no measurable speed
# cost. Memory cost is 32 MB (LITERAL_ORDER3_BUCKETS * 256 * uint16) —
# negligible against the Hutter 10 GB budget and current ~13 MB model state.
# Disable with KOLMO_PPM_ORDER3=0 to compare against historical PPM benches.
_PPM_ORDER3 = os.environ.get("KOLMO_PPM_ORDER3", "1").lower() not in (
    "0", "false", ""
)

# Third predictor: post-copy byte distribution. After a copy event ends, the
# next byte often has a strongly non-uniform distribution conditioned on the
# last byte of the copy — finishing a word, closing a tag, the space after a
# template, etc. PPM doesn't know about copy events at all (it just sees the
# byte stream), so this is genuinely new signal complementary to PPM + neural.
#
# State cost: 256x256 float64 counts = 524 KB per LiteralModel. Negligible.
# Speed cost: one extra blend term whenever the *previous* event was a copy
# (~5-15% of literal predictions in practice).
#
# KOLMO_POST_COPY=1 enables; default 0 until benched. KOLMO_POST_COPY_WEIGHT
# sets its blend weight (0.0 -> ignore post-copy, 1.0 -> only post-copy).
_POST_COPY = os.environ.get("KOLMO_POST_COPY", "0").lower() not in (
    "0", "false", ""
)
LITERAL_POST_COPY_WEIGHT = float(
    os.environ.get("KOLMO_POST_COPY_WEIGHT", "0.15")
)
LITERAL_ORDER4_WEIGHT = 0.20
LITERAL_ORDER4_CONFIDENCE = 2.0
LITERAL_ORDER4_BUCKETS = 1 << 18
LITERAL_ORDER5_WEIGHT = 0.0
LITERAL_ORDER5_CONFIDENCE = 4.0
LITERAL_ORDER5_BUCKETS = 1 << 18
_MASK64 = 0xFFFFFFFFFFFFFFFF


def literal_context_bucket(context: int, buckets: int) -> int:
    """Map a byte context integer to a hashed literal-model bucket.

    The high-order literal tables are bounded and hashed. The old code used
    `context * odd_constant % buckets`; when `buckets` is a power of two, that
    mostly preserves low-bit structure. For byte contexts, low bits are just
    the most recent byte(s), so many distinct order-4 contexts collapsed into
    surprisingly few buckets on enwik prefixes.

    SplitMix64's finalizer gives a cheap avalanche: nearby contexts and
    contexts sharing suffix bytes spread across the whole table. Collisions
    still happen (bounded memory is the point), but they become random noise
    instead of systematic suffix aliasing.
    """
    if buckets <= 0:
        raise ValueError("bucket count must be positive")
    x = (int(context) + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x = (x ^ (x >> 31)) & _MASK64
    if buckets & (buckets - 1) == 0:
        return x & (buckets - 1)
    return x % buckets


# Seed corpus: baked into both encoder and decoder code, costs zero bytes in
# the compressed blob, but trains the model to a useful starting state before
# the user's data is touched. Bigger and more diverse = better prior on common
# English/Wikipedia patterns = fewer bits per literal byte once real data
# starts.
#
# The previous version repeated one paragraph three times, which mostly
# taught the model "this paragraph repeats." Replaced with non-repetitive
# content covering the regimes Hutter (enwik8) actually contains: prose,
# dialogue, lists, tables, references, markup, dates, numbers, code, math.
SEED_CORPUS = (
    # English prose — letter/word/punctuation regularities.
    b"English text is full of small regularities. Letters form words, words "
    b"form phrases, and phrases repeat with punctuation, spacing, and rhythm. "
    b"A compressor that begins from a blank model wastes bits learning that "
    b"spaces are common, vowels follow consonants, and sentences end with a "
    b"period followed by a space and a capital letter. The model should "
    b"already know all of this before the first user byte arrives.\n\n"

    # Wikipedia article-style — title, intro, infobox-ish lines, references.
    b"Compression (information theory)\n\n"
    b"In information theory, data compression is the process of encoding "
    b"information using fewer bits than the original representation. Any "
    b"particular compression is either lossy or lossless. Lossless compression "
    b"reduces bits by identifying and eliminating statistical redundancy, so "
    b"that no information is lost. Lossy compression reduces bits by removing "
    b"unnecessary or less important information.[1]\n\n"
    b"The process of reducing the size of a data file is often referred to as "
    b"data compression. In the context of data transmission, it is called "
    b"source coding: encoding done at the source of the data before it is "
    b"stored or transmitted.[2] Source coding should not be confused with "
    b"channel coding, for error detection and correction, or line coding, "
    b"the means for mapping data onto a signal.\n\n"
    b"See also: entropy (information theory), Kolmogorov complexity, "
    b"arithmetic coding, Huffman coding, Lempel-Ziv-Welch.\n\n"
    b"References\n"
    b"1. Wade, Graham (1994). Signal coding and processing. ISBN 978-0-521-42336-6.\n"
    b"2. Mahdi, O.A.; Mohammed, M.A.; Mohamed, A.J. (November 2012). "
    b"\"Implementing a Novel Approach an Convert Audio Compression to Text Coding "
    b"via Hybrid Technique\". International Journal of Computer Science Issues. 9 "
    b"(6, No. 3): 53-59.\n\n"

    # Wiki markup — links, templates, italics, headers.
    b"== History ==\n\n"
    b"The theoretical basis for compression is provided by [[information theory]] "
    b"and, more specifically, [[Algorithmic information theory|algorithmic "
    b"information theory]] for lossless compression and [[rate-distortion theory]] "
    b"for lossy compression. These fields of study were essentially forged by "
    b"[[Claude Shannon]], who published fundamental papers on the topic in the "
    b"late 1940s and early 1950s. Other topics associated with compression "
    b"include [[coding theory]] and [[statistical inference]].\n\n"
    b"{{Main|Lossless compression}}\n"
    b"Lossless data compression algorithms usually exploit "
    b"[[statistical redundancy]] to represent data without losing any "
    b"[[information]], so that the process is reversible.\n\n"

    # Dialogue — handles colons, names, line breaks.
    b"Dialogue:\n"
    b"Alice: Does the model remember the phrase from earlier in the file?\n"
    b"Ben: It remembered letters and short words, but not the exact sentence.\n"
    b"Alice: Then we need a better prior, a longer context, or an explicit "
    b"copy mechanism for repeated text.\n"
    b"Ben: We already have a copy mechanism. It catches matches above eight "
    b"bytes within a sixty-four-kilobyte window.\n\n"

    # Markdown — lists, code, emphasis.
    b"# Notes on the build\n\n"
    b"- Train deterministically; the encoder and decoder must agree on every "
    b"bit produced.\n"
    b"- Keep the model architecture identical on both sides; any drift in "
    b"weights between compress and decompress breaks the round-trip.\n"
    b"- Measure `gzip`, `kolmo`, ratio, and wall time on every change.\n"
    b"- Revert changes that only help tiny inputs at the cost of large ones.\n"
    b"- The seed corpus is part of the algorithm, not part of the data; it "
    b"costs nothing in the output blob.\n\n"
    b"```python\n"
    b"def compress(data: bytes) -> bytes:\n"
    b"    model = build_model()\n"
    b"    return arithmetic_encode(predict_stream(model, data))\n"
    b"```\n\n"

    # Numbers, dates, units, currencies — common token shapes.
    b"Numbers and dates: 2026-05-22, 1,024 bytes, 2,048 bytes, 4,096 bytes, "
    b"65,536 entries, 10^6 iterations, 3.14159, 2.71828, -273.15 C, 98.6 F, "
    b"$1.99, 49.95 EUR, GBP 12.50, 12:30 PM, 23:59 UTC, 1989-1992, ca. 1850, "
    b"version 1.0.3, RFC 8259, ISO 8601.\n\n"

    # Sentence-level variety — questions, exclamations, parenthetical asides.
    b"Why does a transformer help here? Because text contains both local "
    b"spelling rules (which a small context handles) and long-range reuse "
    b"(which attention captures). A model that handles only local rules will "
    b"plateau; one with useful memory keeps improving as the document grows. "
    b"Note: the model is reset to its seed-warmed state at the start of every "
    b"file, so no information leaks between separate runs.\n\n"

    # Tables — pipes and column structure.
    b"| Algorithm | Type     | Year | Use case               |\n"
    b"|-----------|----------|------|------------------------|\n"
    b"| Huffman   | static   | 1952 | symbol-by-symbol       |\n"
    b"| LZ77      | dictionary | 1977 | general-purpose      |\n"
    b"| Arithmetic | statistical | 1976 | per-bit precision   |\n"
    b"| Neural    | learned  | 2010s | context-sensitive      |\n\n"

    # Math / LaTeX-ish.
    b"Entropy: H(X) = -sum p(x) log p(x), where the base of the logarithm "
    b"determines the unit (bits for log_2, nats for ln). Cross-entropy: "
    b"H(p, q) = -sum p(x) log q(x). The expected code length under arithmetic "
    b"coding equals H(p, q), so a better model q gives shorter blobs. "
    b"KL divergence: D(p || q) = H(p, q) - H(p) >= 0.\n\n"

    # Closing prose — repeats some words from above to reinforce.
    b"A final passage to round out the seed: the city library kept rows of "
    b"shelves, tables, lamps, catalog records, quiet readers, printed forms, "
    b"and old magazines. The same words return in nearby sentences and the "
    b"compressor should pay fewer bits each time a pattern becomes familiar. "
    b"That is the entire point of online learning: shape the distribution to "
    b"match the data as the data arrives.\n\n"

    # ---------------------------------------------------------------------
    # enwik9-targeted patterns (added 2026-06-05). enwik9 is the XML dump of
    # English Wikipedia, so the byte-level statistics it carries are heavy
    # in: XML element tags (<page>, <title>, <text xml:space="preserve">,
    # closing </text></revision></page>, etc.), wiki markup (templates,
    # categories, links, references), biographical/geographic article
    # structure, citation footnote formats, and CC-licence boilerplate.
    # Every byte of seed corpus pays for itself if it saves >1 byte in
    # the final blob — for a 1 GB target, even tiny per-byte savings
    # dwarf the source-code cost of a 10-15 KB seed expansion.
    # ---------------------------------------------------------------------

    # XML element shells — the structural skeleton of every enwik page.
    b"<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" "
    b"xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
    b"xsi:schemaLocation=\"http://www.mediawiki.org/xml/export-0.3/ "
    b"http://www.mediawiki.org/xml/export-0.3.xsd\" version=\"0.3\" "
    b"xml:lang=\"en\">\n  <siteinfo>\n    <sitename>Wikipedia</sitename>\n"
    b"    <base>http://en.wikipedia.org/wiki/Main_Page</base>\n"
    b"    <generator>MediaWiki 1.6alpha</generator>\n"
    b"    <case>first-letter</case>\n      <namespaces>\n"
    b"      <namespace key=\"-2\">Media</namespace>\n"
    b"      <namespace key=\"-1\">Special</namespace>\n"
    b"      <namespace key=\"0\" />\n"
    b"      <namespace key=\"1\">Talk</namespace>\n"
    b"      <namespace key=\"2\">User</namespace>\n"
    b"      <namespace key=\"3\">User talk</namespace>\n"
    b"      <namespace key=\"4\">Wikipedia</namespace>\n"
    b"      <namespace key=\"5\">Wikipedia talk</namespace>\n"
    b"      <namespace key=\"6\">Image</namespace>\n"
    b"      <namespace key=\"7\">Image talk</namespace>\n"
    b"      <namespace key=\"8\">MediaWiki</namespace>\n"
    b"      <namespace key=\"9\">MediaWiki talk</namespace>\n"
    b"      <namespace key=\"10\">Template</namespace>\n"
    b"      <namespace key=\"11\">Template talk</namespace>\n"
    b"      <namespace key=\"12\">Help</namespace>\n"
    b"      <namespace key=\"13\">Help talk</namespace>\n"
    b"      <namespace key=\"14\">Category</namespace>\n"
    b"      <namespace key=\"15\">Category talk</namespace>\n"
    b"    </namespaces>\n  </siteinfo>\n"

    # A typical short article — every tag we expect to see thousands of
    # times in enwik. The text below is original prose written for the
    # seed; it imitates Wikipedia style without quoting any real article.
    b"  <page>\n    <title>Sample article</title>\n    <id>4242</id>\n"
    b"    <revision>\n      <id>1010101</id>\n"
    b"      <timestamp>2004-08-15T14:22:51Z</timestamp>\n"
    b"      <contributor>\n        <username>Anonymous</username>\n"
    b"        <id>0</id>\n      </contributor>\n"
    b"      <comment>initial revision</comment>\n"
    b"      <text xml:space=\"preserve\">"

    # Article body proper — opening sentence, bolded title, dates,
    # lead paragraph, sections, and references. Repeats common
    # function-word transitions ("the", "of", "in", "and").
    b"'''Sample article''' is a short example used to demonstrate the "
    b"structure of a Wikipedia article. It was first written in the early "
    b"2000s and has been edited many times by various contributors. The "
    b"article is intended as a placeholder and does not describe a real "
    b"subject; however, the patterns of words, links, and templates it "
    b"contains are representative of articles that do.\n\n"
    b"== Background ==\nThe article was created on 15 August 2004 by an "
    b"anonymous contributor. Subsequent edits added section headers, "
    b"references, and {{Citation needed}} tags. By 2008, the article had "
    b"grown to roughly one thousand words and contained six references "
    b"to print and online sources.\n\n"
    b"== Structure ==\nMost Wikipedia articles begin with a bolded lead "
    b"sentence and a short summary paragraph, followed by sections marked "
    b"by ''== Section title ==''. Inline citations use the &lt;ref&gt; tag, "
    b"often pointing at the {{Cite book}}, {{Cite journal}}, or "
    b"{{Cite web}} templates defined in [[Template:Citation]].\n\n"
    b"== See also ==\n* [[Main Page]]\n* [[Help:Editing]]\n"
    b"* [[Wikipedia:Manual of Style]]\n\n"
    b"== References ==\n&lt;references/&gt;\n\n"
    b"== External links ==\n* {{Official website|http://example.org/}}\n"
    b"* [http://example.com External example]\n\n"
    b"{{DEFAULTSORT:Sample article}}\n[[Category:Example articles]]\n"
    b"[[Category:Articles created in 2004]]\n[[Category:Stubs]]\n"
    b"{{stub}}"
    b"</text>\n    </revision>\n  </page>\n"

    # Biographical article skeleton — different from generic article in
    # that it has birth/death dates, occupation, life events.
    b"  <page>\n    <title>Jane Smith (mathematician)</title>\n"
    b"    <id>4243</id>\n    <revision>\n      <id>1010102</id>\n"
    b"      <timestamp>2007-03-04T09:11:18Z</timestamp>\n"
    b"      <contributor>\n"
    b"        <username>Editor1</username>\n        <id>12345</id>\n"
    b"      </contributor>\n      <comment>add infobox and references</comment>\n"
    b"      <text xml:space=\"preserve\">"
    b"{{Infobox scientist\n| name              = Jane Smith\n"
    b"| birth_date        = {{birth date|1882|7|9}}\n"
    b"| birth_place       = [[Boston, Massachusetts]], United States\n"
    b"| death_date        = {{death date and age|1968|2|14|1882|7|9}}\n"
    b"| death_place       = [[Cambridge, Massachusetts]], United States\n"
    b"| residence         = United States\n| citizenship       = American\n"
    b"| field             = [[Number theory]], [[combinatorics]]\n"
    b"| work_institution  = [[Massachusetts Institute of Technology]]\n"
    b"| alma_mater        = [[Harvard University]] (PhD, 1908)\n"
    b"| doctoral_advisor  = [[Maxime Bocher]]\n"
    b"| known_for         = Smith's lemma on prime gaps\n"
    b"| prizes            = [[Bocher Memorial Prize]] (1936)\n}}\n"
    b"'''Jane Smith''' (9 July 1882 &ndash; 14 February 1968) was an "
    b"American [[mathematician]] who made contributions to [[number "
    b"theory]] and [[combinatorics]]. She is best remembered for "
    b"Smith's lemma, which gives an effective upper bound on the gap "
    b"between consecutive primes under the [[Riemann hypothesis]].\n\n"
    b"== Early life and education ==\nSmith was born in [[Boston, "
    b"Massachusetts]], the daughter of a printer. She studied at "
    b"[[Radcliffe College]] from 1900 to 1904, then completed her "
    b"doctorate at [[Harvard University]] under [[Maxime Bocher]] in "
    b"1908.&lt;ref name=\"obit\"/&gt; Her dissertation, ''On the "
    b"distribution of quadratic residues modulo a prime'', is still "
    b"occasionally cited.\n\n"
    b"== Career ==\nFrom 1909 until her retirement in 1952 Smith "
    b"taught at the [[Massachusetts Institute of Technology]]. She "
    b"served as departmental chair from 1937 to 1942, was elected to "
    b"the [[American Academy of Arts and Sciences]] in 1929, and was a "
    b"member of the [[National Academy of Sciences]] from 1947 onward."
    b"\n\n== Death ==\nSmith died at her home in Cambridge on 14 "
    b"February 1968 at the age of 85.\n\n"
    b"== Selected publications ==\n"
    b"* {{cite journal|last=Smith|first=Jane|title=On the distribution "
    b"of quadratic residues|journal=Annals of Mathematics|volume=10|"
    b"year=1909|pages=1&ndash;42|jstor=1967392}}\n"
    b"* {{cite book|last=Smith|first=Jane|title=Lectures on number "
    b"theory|publisher=MIT Press|location=Cambridge|year=1948|"
    b"isbn=978-0-262-12345-6}}\n\n"
    b"== References ==\n&lt;references&gt;\n"
    b"&lt;ref name=\"obit\"&gt;{{cite news|title=Obituary: Jane Smith|"
    b"url=http://example.org/obit/1968|newspaper=The Boston Globe|"
    b"date=15 February 1968|page=B12}}&lt;/ref&gt;\n&lt;/references&gt;\n\n"
    b"== External links ==\n"
    b"* {{MacTutor|id=Smith_Jane}}\n"
    b"* {{MathGenealogy|id=42424}}\n\n"
    b"{{Authority control}}\n{{DEFAULTSORT:Smith, Jane}}\n"
    b"[[Category:1882 births]]\n[[Category:1968 deaths]]\n"
    b"[[Category:American mathematicians]]\n"
    b"[[Category:Women mathematicians]]\n"
    b"[[Category:Number theorists]]\n"
    b"[[Category:Harvard University alumni]]\n"
    b"[[Category:Massachusetts Institute of Technology faculty]]\n"
    b"[[Category:People from Boston]]\n"
    b"</text>\n    </revision>\n  </page>\n"

    # Geographic article — town/city style with population, climate,
    # demographics. Different vocab clusters from biography.
    b"  <page>\n    <title>Riverdale, New York</title>\n"
    b"    <id>4244</id>\n    <revision>\n      <id>1010103</id>\n"
    b"      <timestamp>2009-06-12T20:54:30Z</timestamp>\n"
    b"      <contributor>\n        <username>Editor2</username>\n"
    b"        <id>23456</id>\n      </contributor>\n"
    b"      <comment>expand demographics</comment>\n"
    b"      <text xml:space=\"preserve\">"
    b"{{Infobox settlement\n| name             = Riverdale\n"
    b"| settlement_type  = [[Census-designated place]]\n"
    b"| image_skyline    = Riverdale_panorama.jpg\n"
    b"| image_caption    = View from the river bluff\n"
    b"| pushpin_map      = USA New York\n"
    b"| coordinates      = {{coord|40|54|N|73|54|W|display=inline,title}}\n"
    b"| subdivision_type = [[Country|Country]]\n| subdivision_name = "
    b"{{flag|United States}}\n| subdivision_type1 = [[U.S. state|State]]\n"
    b"| subdivision_name1 = {{flag|New York}}\n"
    b"| subdivision_type2 = [[List of counties in New York|County]]\n"
    b"| subdivision_name2 = [[Bronx County, New York|Bronx]]\n"
    b"| area_total_km2   = 4.2\n| population_total = 47,850\n"
    b"| population_as_of = [[2010 United States Census|2010]]\n"
    b"| timezone         = [[Eastern Time Zone|Eastern (EST)]]\n"
    b"| utc_offset       = -5\n| timezone_DST     = EDT\n"
    b"| utc_offset_DST   = -4\n| elevation_m      = 35\n"
    b"| postal_code      = 10463, 10471\n"
    b"| area_code        = [[Area code 718|718]], [[Area code 347|347]]\n}}\n"
    b"'''Riverdale''' is a residential neighborhood in the northwestern "
    b"part of [[the Bronx]], in [[New York City]], [[United States]]. "
    b"It is bounded by [[Yonkers, New York|Yonkers]] to the north, "
    b"[[Van Cortlandt Park]] to the east, [[Kingsbridge, Bronx|"
    b"Kingsbridge]] to the south, and the [[Hudson River]] to the west."
    b"\n\n== Geography ==\nRiverdale sits on a ridge overlooking the "
    b"Hudson River, with elevations ranging from sea level along the "
    b"riverbank to about 80 metres at the top of the bluff. The "
    b"neighborhood is divided informally into North Riverdale, Central "
    b"Riverdale, and Spuyten Duyvil to the south.\n\n"
    b"== Demographics ==\nAccording to the [[United States Census "
    b"Bureau]], the neighborhood had a population of 47,850 at the "
    b"2010 census. The population was 65% [[non-Hispanic whites|White]],"
    b" 14% [[Hispanic and Latino Americans|Hispanic]], 9% [[African "
    b"Americans|Black]], 8% [[Asian Americans|Asian]], and 4% of two "
    b"or more races. The median household income was $74,000.\n\n"
    b"== History ==\nRiverdale was settled in the seventeenth century "
    b"by Dutch farmers. It became part of New York City in 1874 when "
    b"the western Bronx was annexed from [[Westchester County]]. "
    b"The Riverdale-on-Hudson railroad station opened in 1853.\n\n"
    b"== References ==\n{{reflist}}\n\n"
    b"== External links ==\n"
    b"* {{cite web|url=http://riverdale.example.org/|title=Riverdale "
    b"Community Council|access-date=12 June 2009}}\n\n"
    b"{{Bronx neighborhoods}}\n{{DEFAULTSORT:Riverdale, New York}}\n"
    b"[[Category:Neighborhoods in the Bronx]]\n"
    b"[[Category:Hudson River]]\n[[Category:Census-designated places "
    b"in New York]]\n</text>\n    </revision>\n  </page>\n"

    # Disambiguation, redirect, and stub patterns — short pages that
    # occur many times in enwik9 and contain very repetitive structure.
    b"  <page>\n    <title>Smith (disambiguation)</title>\n"
    b"    <id>4245</id>\n    <revision>\n      <id>1010104</id>\n"
    b"      <timestamp>2010-01-01T00:00:00Z</timestamp>\n"
    b"      <contributor><username>BotName</username><id>99</id></contributor>\n"
    b"      <comment>add entries</comment>\n"
    b"      <text xml:space=\"preserve\">"
    b"'''Smith''' is a common English surname; it may also refer to:\n\n"
    b"== People ==\n* [[Smith (surname)]], list of people with the surname\n"
    b"* [[Jane Smith (mathematician)]] (1882&ndash;1968), American "
    b"number theorist\n* [[John Smith (explorer)]] (1580&ndash;1631), "
    b"English colonist of [[Jamestown, Virginia|Jamestown]]\n\n"
    b"== Places ==\n* [[Smith County, Kansas]]\n* [[Smith County, "
    b"Mississippi]]\n* [[Smith County, Tennessee]]\n* [[Smith County, "
    b"Texas]]\n\n== Other ==\n* [[Smith Corona]], a typewriter brand\n"
    b"* [[The Smiths]], an English rock band\n\n"
    b"{{disambiguation|surname}}\n"
    b"</text>\n    </revision>\n  </page>\n"
    b"  <page>\n    <title>Smyth</title>\n    <id>4246</id>\n"
    b"    <revision>\n      <id>1010105</id>\n"
    b"      <timestamp>2003-01-01T00:00:00Z</timestamp>\n"
    b"      <contributor><username>RedirBot</username><id>1</id></contributor>\n"
    b"      <comment>redirect</comment>\n"
    b"      <text xml:space=\"preserve\">"
    b"#REDIRECT [[Smith (surname)]]"
    b"</text>\n    </revision>\n  </page>\n</mediawiki>\n\n"

    # MediaWiki escapes that appear inside <text> — &amp;, &lt;, &gt;,
    # &quot;, &nbsp;, &ndash;, &mdash;. enwik9 has tens of millions.
    b"Common HTML entities and escapes seen inside text bodies include "
    b"&amp;amp;, &amp;lt;, &amp;gt;, &amp;quot;, &amp;nbsp;, &amp;ndash;, "
    b"and &amp;mdash;. URLs containing query strings often look like "
    b"http://example.org/index.php?title=Foo&amp;action=edit&amp;section=2. "
    b"Date ranges are written 1882&amp;ndash;1968 or 14&amp;nbsp;February.\n\n"

    # Cite-template variants — vary by source type (book, journal, web,
    # news, conference). All take similar named parameters.
    b"{{cite book|last=Knuth|first=Donald E.|author-link=Donald Knuth|"
    b"title=The Art of Computer Programming, Volume 1: Fundamental "
    b"Algorithms|edition=3rd|publisher=Addison-Wesley|location=Reading, "
    b"Massachusetts|year=1997|isbn=978-0-201-89683-1|pages=1&ndash;650}}\n"
    b"{{cite journal|last1=Shannon|first1=Claude E.|title=A mathematical "
    b"theory of communication|journal=Bell System Technical Journal|"
    b"volume=27|issue=3|pages=379&ndash;423|year=1948|doi=10.1002/"
    b"j.1538-7305.1948.tb01338.x}}\n"
    b"{{cite web|url=https://example.org/article|title=Sample web page|"
    b"website=Example|access-date=1 January 2020|archive-url=https://web."
    b"archive.org/web/20200101000000/https://example.org/article|"
    b"archive-date=1 January 2020|url-status=live}}\n"
    b"{{cite news|last=Reporter|first=Anne|title=Local council debates "
    b"budget|newspaper=The Daily Example|date=12 March 2019|page=A3|url="
    b"http://example.com/local/12mar2019}}\n"
    b"{{cite conference|last=Researcher|first=Pat|title=A new algorithm "
    b"for arithmetic coding|book-title=Proceedings of the Data "
    b"Compression Conference|publisher=IEEE|year=1992|pages=72&ndash;81}}"
    b"\n\n"

    # Lists, bullets, definition lists — common throughout enwik.
    b"; Term : Definition of the term.\n; Another term : Another definition.\n"
    b"# First numbered item.\n# Second numbered item.\n## Sub-item.\n"
    b"## Another sub-item.\n# Third numbered item.\n* First bullet.\n"
    b"* Second bullet.\n** Indented sub-bullet.\n** Another sub-bullet.\n"
    b"*** Deeper bullet.\n* Third bullet.\n\n"

    # Wikitable — pipe-separated, with class, headers, rows.
    b"{| class=\"wikitable sortable\"\n|+ Sample table\n! Column A !! "
    b"Column B !! Numeric column\n|-\n| First row, first cell || First "
    b"row, second cell || 100\n|-\n| Second row, first cell || Second "
    b"row, second cell || 250\n|-\n| Third row, first cell || Third row,"
    b" second cell || 425\n|}\n\n"

    # CC-BY-SA licence boilerplate fragment.
    b"This article is licensed under the Creative Commons Attribution-"
    b"ShareAlike 3.0 Unported License. Additional terms may apply. By "
    b"using this site, you agree to the Terms of Use and Privacy "
    b"Policy. Wikipedia is a registered trademark of the Wikimedia "
    b"Foundation, Inc., a non-profit organization.\n\n"

    # Talk-page conventions — signatures, indents, headers.
    b"== Discussion of recent edits ==\nI removed the paragraph about "
    b"the 1972 conference because it was unsourced. The source given "
    b"was a personal blog, which does not meet [[WP:RS]]. ~~~~\n"
    b": I agree the source was weak, but I think the underlying claim "
    b"is verifiable. I have added a citation to the original conference "
    b"proceedings, which should resolve the issue. ~~~~\n"
    b":: Thanks. Looks good now. ~~~~\n\n"

    # Final dense lump of common function words and bigrams. Doesn't
    # need to be coherent — what matters is the byte-level bigram and
    # trigram statistics it provides to the order-2/3 PPM tables.
    b"of the and to in a is that for on with as by from at this be or "
    b"are not was it which an have has had will would could should may "
    b"might can also been being do does did about into between through "
    b"during before after above below over under since until against "
    b"because although however therefore moreover furthermore "
    b"consequently nevertheless thus accordingly otherwise instead "
    b"meanwhile elsewhere finally initially originally subsequently "
    b"eventually approximately generally typically often sometimes "
    b"rarely usually currently formerly previously recently historically.\n"
)
EVENT_PROBS = np.array([1.0 - COPY_PROB, COPY_PROB], dtype=np.float64)


@dataclass
class FixedModelState:
    """Fixed-point model state used when KOLMO_FIXED=1."""

    weights: dict[str, np.ndarray]
    optimizer_state: FixedAdamState | None = None
    n_heads: int = 8
    n_layers: int = 4
    use_rope: bool = False
    # Pairs of (canonical, alias) parameter names that share underlying
    # weights — used to sum gradients before Adam and re-alias after.
    tied_params: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tied_params is None:
            self.tied_params = []


def _use_fixed() -> bool:
    return os.environ.get("KOLMO_FIXED", "").lower() in {"1", "true", "yes"}


def _skip_prime() -> bool:
    return os.environ.get("KOLMO_SKIP_PRIME", "").lower() in {"1", "true", "yes"}


def _use_rope() -> bool:
    value = os.environ.get("KOLMO_USE_ROPE")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes"}


# Model presets for hyperparameter sweeps. The "full" preset is the production
# default. The "draft" preset trades ~1pp of ratio for ~2x speed — useful when
# iterating on copy / literal / schedule tuning where the ratio delta between
# configs is what matters, not absolute ratio. Blobs are NOT interchangeable
# across presets; set KOLMO_MODEL on both sides.
_MODEL_PRESETS = {
    # Scale-to-preset matrix (rough rules of thumb; re-bench as data grows):
    #
    #   input scale     recommended preset    rationale
    #   -----------     ------------------    ---------
    #   < 32 KB         draft                 model has too little training
    #                                         data to use bigger capacity;
    #                                         draft trains fastest per byte
    #                                         and is only ~0.014 bpb worse
    #                                         than full at 16 KB enwik9.
    #   32 KB - 1 MB    full (default)        the canonical preset; matches
    #                                         the published bench numbers.
    #   1 MB - 100 MB   large                 hypothesis: 10 M params can
    #                                         actually be filled with signal
    #                                         once we cross 1 MB. Not yet
    #                                         confirmed — the earlier 16 KB
    #                                         large test was WORSE than full
    #                                         due to data starvation. The
    #                                         long-running Windows-GPU enwik9
    #                                         bench is the way to confirm.
    #   > 100 MB        xl (experimental)     for the dev iteration loop
    #                                         only; CPU compute makes this
    #                                         impractical at Hutter scale,
    #                                         so xl is GPU/cloud-only and
    #                                         used to explore "does scaling
    #                                         keep paying off?" not to ship.
    #
    # Blobs are NOT interchangeable across presets; KOLMO_MODEL must match on
    # both compress and decompress.
    "full": dict(d_model=256, n_heads=8, n_layers=4),
    "draft": dict(d_model=192, n_heads=6, n_layers=3),
    # Scaling-law experiment: bigger than full. Earlier 10M-at-4-KB test
    # showed no benefit because there wasn't enough training data; theory
    # says it should win once the model has seen enough bytes to actually
    # use the extra capacity. ~11 M params.
    "large": dict(d_model=384, n_heads=8, n_layers=6),
    # Experimental — for cloud/GPU iteration to test if scaling keeps
    # paying off at very-large data. NOT viable for the Hutter Prize
    # endgame (rules cap at single CPU core / 50 hours; xl is too slow
    # for that). ~26 M params: d_model=512, 8 heads, 8 layers.
    "xl": dict(d_model=512, n_heads=8, n_layers=8, max_context=1024),
}


def _model_preset() -> str:
    name = os.environ.get("KOLMO_MODEL", "full").lower()
    if name not in _MODEL_PRESETS:
        raise ValueError(
            f"unknown KOLMO_MODEL preset {name!r}; "
            f"choices: {sorted(_MODEL_PRESETS)}"
        )
    return name


def offset_probs(n: int) -> np.ndarray:
    """Static prior over offset values 0..n-1 (representing actual offsets
    1..n). Uses 1/sqrt(k) — a reasonable starting point before any events
    are observed. Used by OffsetModel as the initial Laplace prior."""
    if n <= 0:
        return np.array([], dtype=np.float64)
    raw = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float64))
    return raw / math.fsum(raw)


def length_probs(n: int) -> np.ndarray:
    """Probability distribution over length-MIN values 0..n-1. Favors shorter
    matches via 1/k decay — match-length distribution is steeper than offset
    distribution in practice."""
    if n <= 0:
        return np.array([], dtype=np.float64)
    raw = 1.0 / np.arange(1, n + 1, dtype=np.float64)
    return raw / math.fsum(raw)


class LengthModel:
    """Adaptive probability model for copy lengths using log buckets.

    Lengths are represented as offsets from COPY_MIN:

      length_offset = length - COPY_MIN  # 0..COPY_MAX-COPY_MIN

    The old model encoded that offset as one categorical symbol over up to
    1017 choices. That works, but after COPY_MAX=1024 long matches pay for a
    large flat alphabet even though lengths are naturally log-ish: exact 8-byte
    copies are common, then ranges like 9-10, 11-14, 15-22, etc.

    This mirrors OffsetModel:

      1. bucket = floor(log2(length_offset + 1))
      2. residual = length_offset - bucket_lo

    The initial bucket/residual priors are derived from the old 1/k exact
    length prior, so first-copy behavior stays close while the adaptive model
    learns whether a file prefers tiny or long matches.
    """

    def __init__(
        self,
        n: int,
        prior_strength: float = 16.0,
        residual_prior_strength: float = 8.0,
    ):
        self.n = n
        offsets = np.arange(n, dtype=np.int64)
        buckets = np.array(
            [self.bucket_for(int(o)) for o in offsets],
            dtype=np.int64,
        )
        exact_prior = length_probs(n)
        prior = np.bincount(
            buckets,
            weights=exact_prior,
            minlength=n.bit_length(),
        )
        prior = prior / math.fsum(prior) * prior_strength
        self.counts = prior.astype(np.float64)
        self.residual_counts: list[np.ndarray] = []
        for bucket in range(n.bit_length()):
            lo, hi = self.bucket_bounds(bucket, n)
            exact = exact_prior[lo : hi + 1].copy()
            exact = exact / math.fsum(exact) * residual_prior_strength
            self.residual_counts.append(exact.astype(np.float64))

    @staticmethod
    def bucket_for(length_offset: int) -> int:
        if length_offset < 0:
            raise ValueError("length offset must be non-negative")
        return (length_offset + 1).bit_length() - 1

    @staticmethod
    def bucket_bounds(bucket: int, max_n: int) -> tuple[int, int]:
        if bucket < 0:
            raise ValueError("bucket must be non-negative")
        if max_n <= 0:
            raise ValueError("max_n must be positive")
        lo = (1 << bucket) - 1
        hi = min((1 << (bucket + 1)) - 2, max_n - 1)
        if lo > hi:
            raise ValueError("bucket is not legal for max_n")
        return lo, hi

    def probs_for(self, max_n: int) -> np.ndarray:
        """Return normalized probabilities over legal length buckets."""
        if max_n <= 0:
            return np.array([], dtype=np.float64)
        p = self.counts[: max_n.bit_length()].copy()
        return p / math.fsum(p)

    def residual_probs_for(self, bucket: int, max_n: int) -> np.ndarray:
        lo, hi = self.bucket_bounds(bucket, max_n)
        width = hi - lo + 1
        p = self.residual_counts[bucket][:width].copy()
        return p / math.fsum(p)

    def observe(self, length_offset: int) -> None:
        bucket = self.bucket_for(length_offset)
        lo, _ = self.bucket_bounds(bucket, self.n)
        self.counts[bucket] += 1.0
        self.residual_counts[bucket][length_offset - lo] += 1.0


class EventModel:
    """Adaptive probability model for the literal/copy event flag.

    The fixed EVENT_PROBS = [0.995, 0.005] assumes a 0.5% copy rate, but real
    text shows 5-15% rates once enough history is available. This costs ~7.6
    bits per copy flag with the static prior; with adaptation, copies in long
    files cost ~3 bits.

    Both encoder and decoder hold an instance and call `observe` after every
    event, in the same order — distribution evolves bit-identically.
    """

    def __init__(self, prior_copy: float = 0.05, prior_strength: float = 50.0):
        self.copy_count = prior_copy * prior_strength
        self.literal_count = (1.0 - prior_copy) * prior_strength

    def probs(self) -> np.ndarray:
        total = self.copy_count + self.literal_count
        return np.array(
            [self.literal_count / total, self.copy_count / total],
            dtype=np.float64,
        )

    def observe(self, event: int) -> None:
        if event == 1:
            self.copy_count += 1.0
        else:
            self.literal_count += 1.0


class LiteralModel:
    """Adaptive byte-context model mixed with neural literal probabilities.

    The transformer adapts via comparatively expensive optimizer steps. This
    model adapts immediately after every observed byte, including bytes emitted
    by copy events, and captures cheap file-local regularities such as:
    - after '<' in wiki/XML markup, letters and '/' are common
    - after '[' another '[' is common
    - after '\n' markup bullets, headings, and capitals are common

    It is deliberately bounded: order-0 counts, a dense order-1 table, a dense
    order-2 table, and optional hashed order-3/order-4/order-5 tables. Dense
    exact order-3 would be too large (2^24 contexts * 256 next bytes), and
    dense exact order-4/order-5 is completely out. The hashed tables are
    fixed-size and collisions only smear the distribution.
    """

    def __init__(self, prior: float = 1.0):
        self.count0 = np.full(256, prior, dtype=np.float64)
        self.count1 = np.full((256, 256), prior, dtype=np.float64)
        self.count2 = np.zeros((256 * 256, 256), dtype=np.uint32)
        self.count3 = (
            np.zeros((LITERAL_ORDER3_BUCKETS, 256), dtype=np.uint16)
            if (LITERAL_ORDER3_WEIGHT > 0.0 or _PPM_ORDER3)
            else None
        )
        self.count4 = (
            np.zeros((LITERAL_ORDER4_BUCKETS, 256), dtype=np.uint16)
            if LITERAL_ORDER4_WEIGHT > 0.0
            else None
        )
        self.count5 = (
            np.zeros((LITERAL_ORDER5_BUCKETS, 256), dtype=np.uint16)
            if LITERAL_ORDER5_WEIGHT > 0.0
            else None
        )
        self.prev5 = BOS
        self.prev4 = BOS
        self.prev3 = BOS
        self.prev2 = BOS
        self.prev = BOS

        # Post-copy predictor state (third predictor, see KOLMO_POST_COPY).
        # `post_copy_counts[last_byte_of_copy, next_byte_observed]` is
        # incremented after the next literal observe immediately following
        # `mark_copy_end()`. Prior 1.0 so an unseen (last, next) gives a
        # uniform-with-weight-1 distribution before any observations.
        self.post_copy_counts = np.full((256, 256), 1.0, dtype=np.float64)
        # When True, the *next* call to observe() should record into
        # post_copy_counts[_last_copy_end_byte, byte] before clearing.
        self._after_copy = False
        self._last_copy_end_byte = 0

    def mark_copy_end(self, last_byte: int) -> None:
        """Tell the model the most recent event was a copy whose last byte
        was `last_byte`. The next observe() will be recorded as a post-copy
        transition, and the next probs() call will blend in the conditional
        post-copy distribution. No-op if the post-copy predictor is disabled,
        but the bookkeeping is cheap so we always do it.
        """
        self._after_copy = True
        self._last_copy_end_byte = int(last_byte)

    def _post_copy_distribution(self) -> np.ndarray | None:
        """Conditional distribution over the next byte given that the
        previous event was a copy ending in `_last_copy_end_byte`. Returns
        None if we're not in a post-copy state (so the caller knows to skip
        the blend term)."""
        if not self._after_copy:
            return None
        row = self.post_copy_counts[self._last_copy_end_byte]
        return row / math.fsum(row)

    def _ppm_distribution(self) -> np.ndarray:
        """PPM-C blended byte distribution.

        Walks orders 5, 4, 3, 2, 1, 0 from longest to shortest. Orders
        whose count tables are None (weight = 0) are skipped. At each
        order:
          - p(b) = count[b] / (sum + distinct)  for b seen at this order
          - escape = distinct / (sum + distinct)
        A byte's final probability is the first (longest-context) match
        times all the escapes above it. Bytes that never appear at any
        order get a tiny uniform 1/256 share of the remaining escape.

        This is the principled way to combine variable-order context
        counts — every byte pays its actual escape cost only for the
        orders that didn't match, instead of paying the static blend
        every time like the legacy "mix" strategy.
        """
        p = np.zeros(256, dtype=np.float64)
        accounted = np.zeros(256, dtype=bool)
        escape = 1.0

        def fold(row: np.ndarray) -> None:
            nonlocal escape
            # row is float64 of small exact-integer counts; sum is small enough
            # to be exact regardless of order, but fsum makes that guarantee
            # version-independent. seen.sum() is a bool reduction → integer
            # count, already deterministic.
            s = math.fsum(row)
            if s <= 0.0:
                return
            seen = row > 0.0
            distinct = float(seen.sum())
            denom = s + distinct
            mask = seen & ~accounted
            p[mask] += escape * row[mask] / denom
            accounted[seen] = True
            escape *= distinct / denom

        # Order 5 (hashed) — only walked if user has enabled it.
        if self.count5 is not None:
            ctx5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            fold(
                self.count5[
                    literal_context_bucket(ctx5, LITERAL_ORDER5_BUCKETS)
                ].astype(np.float64)
            )

        # Order 4 (hashed)
        if self.count4 is not None:
            ctx4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            fold(
                self.count4[
                    literal_context_bucket(ctx4, LITERAL_ORDER4_BUCKETS)
                ].astype(np.float64)
            )

        # Order 3 (hashed) — disabled by default.
        if self.count3 is not None:
            ctx3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            fold(
                self.count3[
                    literal_context_bucket(ctx3, LITERAL_ORDER3_BUCKETS)
                ].astype(np.float64)
            )

        # Order 2 (dense)
        ctx2 = (self.prev2 << 8) | self.prev
        fold(self.count2[ctx2].astype(np.float64))

        # Order 1 (dense, float counts include prior)
        fold(self.count1[self.prev])

        # Order 0 (float counts always positive due to prior=1.0)
        fold(self.count0)

        # Anything still unaccounted (only possible if all orders had
        # s=0, which can't happen because order 0 always has prior=1.0).
        # Spread remaining escape uniformly as a defensive fallback.
        unaccounted = ~accounted
        n_unaccounted = int(unaccounted.sum())
        if n_unaccounted > 0:
            p[unaccounted] += escape / 256.0

        # Use math.fsum for the final renorm: numpy's vectorized .sum()
        # gives platform-dependent results in the last ULP across SIMD
        # widths and numpy versions, which propagates through float
        # division and ultimately changes the int frequencies passed to
        # the arithmetic coder — a Rung-4 (cross-platform) bug. fsum is
        # the correctly-rounded float sum, order-independent, stdlib.
        return p / math.fsum(p)

    def probs(self, neural_probs: np.ndarray) -> np.ndarray:
        if _LITERAL_STRATEGY == "ppm":
            p_neural = neural_probs.astype(np.float64, copy=False)
            # fsum, not np.sum: see comment at end of _ppm_distribution.
            n_sum = math.fsum(p_neural)
            if n_sum > 0.0:
                p_neural = p_neural / n_sum
            p_ppm = self._ppm_distribution()
            if _ADAPTIVE_WEIGHT:
                # Cost-aware blend: use max(p_ppm) as a PPM confidence proxy.
                # peak_norm = 0 when PPM is uniform (1/256 spread), 1 when it
                # collapses onto a single byte. We then interpolate the neural
                # weight between HIGH (peak_norm=0) and LOW (peak_norm=1).
                # float() because np.max returns a numpy scalar; arithmetic on
                # Python floats is identical across platforms via IEEE-754.
                peak = float(p_ppm.max())
                peak_norm = (peak - 1.0 / 256.0) / (1.0 - 1.0 / 256.0)
                if peak_norm < 0.0:
                    peak_norm = 0.0
                elif peak_norm > 1.0:
                    peak_norm = 1.0
                w = (
                    LITERAL_NEURAL_WEIGHT_HIGH * (1.0 - peak_norm)
                    + LITERAL_NEURAL_WEIGHT_LOW * peak_norm
                )
            else:
                w = LITERAL_NEURAL_WEIGHT
            # Optional 3-way blend with post-copy distribution. Active only
            # right after a copy event and only if KOLMO_POST_COPY=1. We
            # scale the post-copy weight down proportionally from the PPM
            # side so neural's weight is unchanged — the intuition is that
            # PPM and post-copy are both "structural" signals that the
            # neural model partly overlaps; the neural model is independent.
            if _POST_COPY:
                post = self._post_copy_distribution()
                if post is not None:
                    pc = LITERAL_POST_COPY_WEIGHT
                    if pc > 1.0 - w:
                        pc = 1.0 - w
                    w_ppm = 1.0 - w - pc
                    mixed = w * p_neural + w_ppm * p_ppm + pc * post
                    return mixed / math.fsum(mixed)
            mixed = w * p_neural + (1.0 - w) * p_ppm
            return mixed / math.fsum(mixed)

        p = neural_probs.astype(np.float64, copy=True)
        if (
            LITERAL_ORDER0_WEIGHT <= 0.0
            and LITERAL_ORDER1_WEIGHT <= 0.0
            and LITERAL_ORDER2_WEIGHT <= 0.0
            and LITERAL_ORDER3_WEIGHT <= 0.0
            and LITERAL_ORDER4_WEIGHT <= 0.0
            and LITERAL_ORDER5_WEIGHT <= 0.0
        ):
            return p / math.fsum(p)

        order5_w = 0.0
        p5 = p
        if self.count5 is not None:
            context5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
            row5 = self.count5[bucket5]
            row5_sum = int(row5.sum())
            if row5_sum > 0:
                p5 = row5.astype(np.float64) / float(row5_sum)
                confidence5 = (
                    row5_sum / (row5_sum + LITERAL_ORDER5_CONFIDENCE)
                    if LITERAL_ORDER5_CONFIDENCE > 0.0
                    else 1.0
                )
                order5_w = LITERAL_ORDER5_WEIGHT * confidence5

        order4_w = 0.0
        p4 = p
        if self.count4 is not None:
            context4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
            row4 = self.count4[bucket4]
            row4_sum = int(row4.sum())
            if row4_sum > 0:
                p4 = row4.astype(np.float64) / float(row4_sum)
                confidence4 = (
                    row4_sum / (row4_sum + LITERAL_ORDER4_CONFIDENCE)
                    if LITERAL_ORDER4_CONFIDENCE > 0.0
                    else 1.0
                )
                order4_w = LITERAL_ORDER4_WEIGHT * confidence4

        order3_w = 0.0
        p3 = p
        if self.count3 is not None:
            context3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            bucket3 = literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
            row3 = self.count3[bucket3]
            row3_sum = int(row3.sum())
            if row3_sum > 0:
                p3 = row3.astype(np.float64) / float(row3_sum)
                confidence3 = (
                    row3_sum / (row3_sum + LITERAL_ORDER3_CONFIDENCE)
                    if LITERAL_ORDER3_CONFIDENCE > 0.0
                    else 1.0
                )
                order3_w = LITERAL_ORDER3_WEIGHT * confidence3

        # count0/count1 hold exact-integer float64 values whose sum is small
        # enough to be exact in float64 regardless of summation order, so
        # np.sum here is *probably* deterministic — but let's not depend on
        # numpy's vectorized reduction being conservative across versions.
        p0 = self.count0 / math.fsum(self.count0)
        row = self.count1[self.prev]
        p1 = row / math.fsum(row)
        context2 = (self.prev2 << 8) | self.prev
        row2 = self.count2[context2]
        row2_sum = int(row2.sum())
        if row2_sum > 0:
            p2 = row2.astype(np.float64) / float(row2_sum)
            if LITERAL_ORDER2_CONFIDENCE > 0.0:
                confidence = row2_sum / (row2_sum + LITERAL_ORDER2_CONFIDENCE)
                order2_w = LITERAL_ORDER2_WEIGHT * confidence
            else:
                order2_w = LITERAL_ORDER2_WEIGHT
        else:
            p2 = p
            order2_w = 0.0
        neural_w = max(
            0.0,
            1.0
            - LITERAL_ORDER0_WEIGHT
            - LITERAL_ORDER1_WEIGHT
            - order2_w
            - order3_w
            - order4_w
            - order5_w,
        )
        mixed = (
            neural_w * p
            + LITERAL_ORDER0_WEIGHT * p0
            + LITERAL_ORDER1_WEIGHT * p1
            + order2_w * p2
            + order3_w * p3
            + order4_w * p4
            + order5_w * p5
        )
        return mixed / math.fsum(mixed)

    def observe(self, byte: int) -> None:
        # Record post-copy transition if the previous event was a copy.
        # We always do the bookkeeping (even when KOLMO_POST_COPY=0) — it's
        # one float increment, cheaper than the env-var read it'd take to
        # gate. The blend itself is what's gated, in probs().
        if self._after_copy:
            self.post_copy_counts[self._last_copy_end_byte, byte] += 1.0
            self._after_copy = False
        self.count0[byte] += 1.0
        self.count1[self.prev, byte] += 1.0
        context2 = (self.prev2 << 8) | self.prev
        self.count2[context2, byte] += 1
        if self.count3 is not None:
            context3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            bucket3 = literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
            if self.count3[bucket3, byte] < np.iinfo(np.uint16).max:
                self.count3[bucket3, byte] += 1
        if self.count4 is not None:
            context4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
            if self.count4[bucket4, byte] < np.iinfo(np.uint16).max:
                self.count4[bucket4, byte] += 1
        if self.count5 is not None:
            context5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
            if self.count5[bucket5, byte] < np.iinfo(np.uint16).max:
                self.count5[bucket5, byte] += 1
        self.prev5 = self.prev4
        self.prev4 = self.prev3
        self.prev3 = self.prev2
        self.prev2 = self.prev
        self.prev = byte

    def proxy_bits(self, seq: bytes | bytearray, neural_bpb: float) -> float:
        """Cheap estimate of literal bits for a known byte sequence.

        Used only by the encoder when deciding whether a copy candidate is
        worth its header. We don't have future neural probabilities without
        actually stepping the transformer through the candidate, so this uses
        `neural_bpb` as a constant proxy for the neural component and adds the
        current adaptive byte-context probabilities for the actual bytes.

        The method intentionally does not mutate counts. It advances the local
        context variables while reading the current tables; that is enough to
        distinguish "the byte model expects this sequence" from "copy header is
        probably cheaper" without spending GPU work on rejected candidates.
        """
        if not seq:
            return 0.0
        base_p = 2.0 ** (-neural_bpb)
        count0_sum = math.fsum(self.count0)
        prev5 = self.prev5
        prev4 = self.prev4
        prev3 = self.prev3
        prev2 = self.prev2
        prev = self.prev
        total_bits = 0.0
        for byte in seq:
            p0 = float(self.count0[byte] / count0_sum)
            row1 = self.count1[prev]
            p1 = float(row1[byte] / math.fsum(row1))

            context2 = (prev2 << 8) | prev
            row2 = self.count2[context2]
            row2_sum = int(row2.sum())
            if row2_sum > 0:
                p2 = float(row2[byte] / row2_sum)
                if LITERAL_ORDER2_CONFIDENCE > 0.0:
                    confidence2 = row2_sum / (row2_sum + LITERAL_ORDER2_CONFIDENCE)
                    order2_w = LITERAL_ORDER2_WEIGHT * confidence2
                else:
                    order2_w = LITERAL_ORDER2_WEIGHT
            else:
                p2 = base_p
                order2_w = 0.0

            order5_w = 0.0
            p5 = base_p
            if self.count5 is not None:
                context5 = (
                    (prev5 << 32)
                    | (prev4 << 24)
                    | (prev3 << 16)
                    | (prev2 << 8)
                    | prev
                )
                bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
                row5 = self.count5[bucket5]
                row5_sum = int(row5.sum())
                if row5_sum > 0:
                    p5 = float(row5[byte] / row5_sum)
                    confidence5 = (
                        row5_sum / (row5_sum + LITERAL_ORDER5_CONFIDENCE)
                        if LITERAL_ORDER5_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order5_w = LITERAL_ORDER5_WEIGHT * confidence5

            order4_w = 0.0
            p4 = base_p
            if self.count4 is not None:
                context4 = (
                    (prev4 << 24) | (prev3 << 16) | (prev2 << 8) | prev
                )
                bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
                row4 = self.count4[bucket4]
                row4_sum = int(row4.sum())
                if row4_sum > 0:
                    p4 = float(row4[byte] / row4_sum)
                    confidence4 = (
                        row4_sum / (row4_sum + LITERAL_ORDER4_CONFIDENCE)
                        if LITERAL_ORDER4_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order4_w = LITERAL_ORDER4_WEIGHT * confidence4

            order3_w = 0.0
            p3 = base_p
            if self.count3 is not None:
                context3 = (prev3 << 16) | (prev2 << 8) | prev
                bucket3 = (
                    literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
                )
                row3 = self.count3[bucket3]
                row3_sum = int(row3.sum())
                if row3_sum > 0:
                    p3 = float(row3[byte] / row3_sum)
                    confidence3 = (
                        row3_sum / (row3_sum + LITERAL_ORDER3_CONFIDENCE)
                        if LITERAL_ORDER3_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order3_w = LITERAL_ORDER3_WEIGHT * confidence3

            neural_w = max(
                0.0,
                1.0
                - LITERAL_ORDER0_WEIGHT
                - LITERAL_ORDER1_WEIGHT
                - order2_w
                - order3_w
                - order4_w
                - order5_w,
            )
            p = (
                neural_w * base_p
                + LITERAL_ORDER0_WEIGHT * p0
                + LITERAL_ORDER1_WEIGHT * p1
                + order2_w * p2
                + order3_w * p3
                + order4_w * p4
                + order5_w * p5
            )
            total_bits += -np.log2(max(p, 1e-300))
            prev5 = prev4
            prev4 = prev3
            prev3 = prev2
            prev2 = prev
            prev = int(byte)
        return float(total_bits)


class OffsetModel:
    """Adaptive probability model for copy offset log-buckets.

    Both compress and decompress hold an instance and call `observe` after
    every copy event, in the same order with the same offsets — so the
    distribution evolves bit-identically on both sides.

    Encoding an exact offset in a 64 KB window as one categorical symbol is
    expensive: every copy event builds a 65,536-way model, and rare long
    offsets pay for a giant alphabet. Instead, encode:

      1. bucket = floor(log2(offset)) with adaptive bucket probabilities
      2. residual = offset - 2^bucket with adaptive within-bucket counts

    This is gzip-style distance coding. The initial bucket prior is derived
    from the old 1/sqrt(offset) prior by summing that mass into buckets, so
    the first-copy behavior remains sensible while the alphabet shrinks from
    65,536 symbols to at most 17 for the first stage. Residual priors are also
    initialized from 1/sqrt(offset), so the initial factorized probability is
    close to the old exact-offset prior while still allowing common exact
    offsets to become cheap.
    """

    def __init__(
        self,
        window: int,
        prior_strength: float = 128.0,
        residual_prior_strength: float = 16.0,
    ):
        self.window = window
        offsets = np.arange(1, window + 1, dtype=np.int64)
        buckets = np.array([self.bucket_for(int(o)) for o in offsets], dtype=np.int64)
        raw = 1.0 / np.sqrt(offsets.astype(np.float64))
        prior = np.bincount(buckets, weights=raw, minlength=window.bit_length())
        prior = prior / math.fsum(prior) * prior_strength
        self.counts = prior.astype(np.float64)
        self.residual_counts: list[np.ndarray] = []
        for bucket in range(window.bit_length()):
            lo, hi = self.bucket_bounds(bucket, window)
            bucket_offsets = np.arange(lo, hi + 1, dtype=np.float64)
            residual_prior = 1.0 / np.sqrt(bucket_offsets)
            residual_prior = (
                residual_prior
                / math.fsum(residual_prior)
                * residual_prior_strength
            )
            self.residual_counts.append(residual_prior.astype(np.float64))

    def probs_for(self, max_offset: int) -> np.ndarray:
        """Return normalized probabilities over legal offset buckets."""
        if max_offset <= 0:
            return np.array([], dtype=np.float64)
        p = self.counts[: max_offset.bit_length()].copy()
        return p / math.fsum(p)

    @staticmethod
    def bucket_for(offset: int) -> int:
        if offset <= 0:
            raise ValueError("copy offset must be positive")
        return offset.bit_length() - 1

    @staticmethod
    def bucket_bounds(bucket: int, max_offset: int) -> tuple[int, int]:
        if bucket < 0:
            raise ValueError("bucket must be non-negative")
        lo = 1 << bucket
        hi = min((1 << (bucket + 1)) - 1, max_offset)
        if lo > hi:
            raise ValueError("bucket is not legal for max_offset")
        return lo, hi

    def residual_probs_for(self, bucket: int, max_offset: int) -> np.ndarray:
        lo, hi = self.bucket_bounds(bucket, max_offset)
        width = hi - lo + 1
        p = self.residual_counts[bucket][:width].copy()
        return p / math.fsum(p)

    def observe(self, offset: int) -> None:
        """Record an offset observation by bucket and residual."""
        bucket = self.bucket_for(offset)
        lo, _ = self.bucket_bounds(bucket, self.window)
        self.counts[bucket] += 1.0
        self.residual_counts[bucket][offset - lo] += 1.0


def _select_device() -> torch.device:
    """Pick CUDA when available so per-byte forward/backward runs on GPU.

    Determinism caveat: GPU ops are non-deterministic across machines, so
    cross-machine round-trip will diverge. For Rung 1 (single-machine) this
    is fine; Rung 2 is where we drop PyTorch entirely for bit-identical
    cross-platform output.

    Override with KOLMO_DEVICE=cpu to force CPU.
    """
    forced = os.environ.get("KOLMO_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def new_model_and_optimizer() -> tuple[KolmoTransformer | FixedModelState, torch.optim.Optimizer | None]:
    """Build a model with deterministic init. Both compress and decompress
    must call this and get bit-identical starting weights."""
    torch.manual_seed(SEED)
    use_rope = _use_rope()
    preset_kwargs = dict(_MODEL_PRESETS[_model_preset()])
    # max_context must exceed the highest absolute position ever indexed.
    # After warm_cache (positions 0..len(history)-1, len(history) <= CONTEXT),
    # step_cache increments pos_offset per byte. A training block can grow
    # up to training_block_size_at == min(BLOCK_SIZE * mult, CONTEXT - 1)
    # before firing, so the worst case is 2*CONTEXT - 1 (full history +
    # full pending). Round up to a power of two for clean RoPE tables.
    min_max_context = 2 * CONTEXT
    max_context = 512
    while max_context < min_max_context:
        max_context *= 2
    preset_kwargs.setdefault("max_context", max_context)
    model = KolmoTransformer(use_rope=use_rope, **preset_kwargs)
    stable_init_model(model, SEED)
    if _use_fixed():
        fixed_model = FixedModelState(
            weights=extract_fixed_weights(model),
            tied_params=tied_param_pairs(model),
            use_rope=use_rope,
        )
        if not _skip_prime():
            if _load_primed_state(fixed_model, model):
                return fixed_model, None
            _prime_model(fixed_model, None)
            _save_primed_state(fixed_model, model)
        return fixed_model, None
    model.to(_select_device())
    model.train()
    if os.environ.get("KOLMO_TORCH_COMPILE", "0").lower() not in ("0", "false", ""):
        # Opt-in torch.compile. The profile shows ~3.7 s spent in
        # torch._C._nn.linear on a 2 KB compress, almost all of it small-
        # matmul dispatch overhead — exactly what compile fuses away.
        # Default OFF because: (1) determinism guarantees are PyTorch-mode-
        # only and we haven't proven the compiled graph is bit-identical
        # across machines; (2) the first call triggers a compile pass that
        # can take several seconds, which is annoying for the tiny-payload
        # regression tests; (3) Dynamo isn't supported on every (torch,
        # python) pair — torch 2.2 on Python 3.12 raises at compile time.
        # Toggle with KOLMO_TORCH_COMPILE=1.
        try:
            model = torch.compile(model)
        except RuntimeError as exc:
            import warnings
            warnings.warn(
                f"KOLMO_TORCH_COMPILE=1 but torch.compile failed "
                f"({exc!r}); falling back to eager mode",
                stacklevel=2,
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if not _skip_prime():
        _prime_model(model, optimizer)
    return model, optimizer


def _prime_model(
    model: KolmoTransformer | FixedModelState,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    """Train on a tiny built-in corpus before real data starts."""
    history = [BOS]
    for pos in range(0, len(SEED_CORPUS), BLOCK_SIZE):
        block = list(SEED_CORPUS[pos : pos + BLOCK_SIZE])
        train_block(model, optimizer, history, block)
        history = update_history(history, block)


def _seed_cache_config(model: KolmoTransformer) -> dict:
    """The bits of model architecture that affect the primed state."""
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.blocks[0].attn.n_heads,
        "n_layers": len(model.blocks),
        "max_context": model.max_context,
        "tie_weights": model.tie_weights,
        "use_rope": model.use_rope,
        "lr": LR,
        "context": CONTEXT,
        "bos": BOS,
    }


def _load_primed_state(
    fixed_model: FixedModelState,
    pytorch_model: KolmoTransformer,
) -> bool:
    """Try to load the primed state from disk. Returns True on hit."""
    from kolmo.seed_cache import (
        cache_disabled,
        cache_path_for,
        compute_config_hash,
        load_state,
    )

    if cache_disabled():
        return False
    config_hash = compute_config_hash(
        seed_corpus=SEED_CORPUS,
        model_config=_seed_cache_config(pytorch_model),
        init_seed=SEED,
        block_size=BLOCK_SIZE,
    )
    path = cache_path_for(config_hash)
    if not path.exists():
        return False
    weights, state, tied = load_state(path)
    fixed_model.weights = weights
    fixed_model.optimizer_state = state
    fixed_model.tied_params = tied
    return True


def _save_primed_state(
    fixed_model: FixedModelState,
    pytorch_model: KolmoTransformer,
) -> None:
    """Save the primed state to disk. Quiet on failure — the cache is an
    optimization, not a correctness requirement."""
    from kolmo.seed_cache import (
        cache_disabled,
        cache_path_for,
        compute_config_hash,
        save_state,
    )

    if cache_disabled() or fixed_model.optimizer_state is None:
        return
    config_hash = compute_config_hash(
        seed_corpus=SEED_CORPUS,
        model_config=_seed_cache_config(pytorch_model),
        init_seed=SEED,
        block_size=BLOCK_SIZE,
    )
    path = cache_path_for(config_hash)
    try:
        save_state(
            path,
            fixed_model.weights,
            fixed_model.optimizer_state,
            fixed_model.tied_params,
        )
    except OSError:
        # Disk full, permission denied, etc. — we already primed in memory,
        # so the current run succeeds; subsequent runs will just re-prime.
        pass


def _trim_caches(caches: list, max_len: int) -> list:
    """Slide the KV cache window: keep only the last `max_len` positions."""
    out = []
    for c in caches:
        if c["k"].shape[2] > max_len:
            out.append({
                "k": c["k"][:, :, -max_len:],
                "v": c["v"][:, :, -max_len:],
            })
        else:
            out.append(c)
    return out


def warm_cache(model: KolmoTransformer, history: list[int]) -> tuple[np.ndarray, list, int]:
    """Run a fresh forward over `history` (no grad) to rebuild the KV cache
    and get the prediction for the next byte. Used at the start of each block,
    after a training step has invalidated the previous cache.

    Returns (probs over next byte as float64 numpy, kv_caches, pos_after).
    """
    if isinstance(model, FixedModelState):
        last_logits_q, caches = fixed_warm(
            np.array(history, dtype=np.int64),
            model.weights,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            pos_offset=0,
            use_rope=model.use_rope,
        )
        # If the prime/seed history is already longer than CONTEXT, the
        # warmed cache exceeds the window — trim now so subsequent steps
        # operate on the same window the PyTorch path would.
        if caches and caches[0]["k"].shape[1] > CONTEXT:
            caches = trim_caches(caches, CONTEXT)
        probs = _probs_from_q15_logits(last_logits_q)
        return probs, caches, len(history)

    device = next(model.parameters()).device
    x = torch.tensor([history], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=None, pos_offset=0)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    # Quantize through deterministic int frequencies so probs derived
    # from float math on different machines collapse to the same values.
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, len(history)


def step_cache(
    model: KolmoTransformer | FixedModelState,
    byte: int,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed one new byte using the cache. Returns (probs over next byte,
    updated caches, new pos_offset)."""
    if isinstance(model, FixedModelState):
        last_logits_q, caches = fixed_step(
            byte,
            caches,
            model.weights,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            pos_offset=pos_offset,
            use_rope=model.use_rope,
        )
        if caches and caches[0]["k"].shape[1] > CONTEXT:
            caches = trim_caches(caches, CONTEXT)
        probs = _probs_from_q15_logits(last_logits_q)
        return probs, caches, pos_offset + 1

    device = next(model.parameters()).device
    x = torch.tensor([[byte]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, pos_offset + 1


def step_cache_batch(
    model: KolmoTransformer | FixedModelState,
    bytes_list: list[int] | bytes | bytearray,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed a sequence of new bytes through the KV cache in one forward pass.

    Used for copy events, where the bytes are already known (from the copy's
    offset+length) so per-byte probabilities are not needed for encoding.
    The cache still has to absorb all N bytes so the next prediction is
    accurate. One forward over N tokens is much faster than N forwards over
    1 token because matmul efficiency scales with the batch dim.

    Returns (probs for the byte AFTER the batch, updated caches, new pos_offset).
    The returned `probs` is the same as if the last byte's `step_cache` had
    been called individually.
    """
    if not bytes_list:
        # No-op convenience; callers typically guarantee non-empty.
        return np.zeros(0, dtype=np.float64), caches, pos_offset

    if isinstance(model, FixedModelState):
        # Fixed mode doesn't have a batched step yet; fall back to a per-byte
        # loop. Still saves the function-call overhead vs the outer caller
        # doing the loop, and keeps the interface uniform.
        last_probs = None
        for byte in bytes_list:
            last_probs, caches, pos_offset = step_cache(
                model, int(byte), caches, pos_offset
            )
        return last_probs, caches, pos_offset

    device = next(model.parameters()).device
    x = torch.tensor([list(bytes_list)], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, pos_offset + len(bytes_list)


def _probs_from_q15_logits(last_logits_q: np.ndarray) -> np.ndarray:
    """Dequantize Q15 logits and quantize them through the deterministic
    int-frequency grid so the resulting probs match the PyTorch path."""
    last_logits = dequantize(last_logits_q).astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    return freqs.astype(np.float64) / float(TOTAL_FREQ)


def train_block(
    model: KolmoTransformer | FixedModelState,
    optimizer: torch.optim.Optimizer | None,
    history: list[int],
    block_bytes: list[int],
) -> None:
    """Run a full forward with gradient over `history + block_bytes`, compute
    cross-entropy loss against the block targets, backward + optimizer step.

    Both compress and decompress call this with the same arguments at the
    same step, so weights stay in lockstep.
    """
    if isinstance(model, FixedModelState):
        model.optimizer_state = fixed_train_block(
            model.weights,
            model.optimizer_state,
            history,
            block_bytes,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            context=CONTEXT,
            tied_params=model.tied_params,
            use_rope=model.use_rope,
        )
        return

    if optimizer is None:
        raise ValueError("PyTorch training requires an optimizer")

    full = (history + block_bytes)[-CONTEXT:]
    m = len(block_bytes)
    n_hist = len(full) - m

    device = next(model.parameters()).device
    x = torch.tensor([full], dtype=torch.long, device=device)
    logits, _ = model(x, kv_caches=None, pos_offset=0)
    # Predictions for block bytes come from logits at positions [n_hist-1 .. n_hist+m-2]
    block_logits = logits[0, n_hist - 1 : n_hist + m - 1]

    targets = torch.tensor(block_bytes, dtype=torch.long, device=device)
    loss = F.cross_entropy(block_logits, targets)
    optimizer.zero_grad()
    loss.backward()
    # Linear LR warmup. Stored on the optimizer (no separate counter).
    step_num = getattr(optimizer, "_kolmo_step", 0) + 1
    optimizer._kolmo_step = step_num
    if step_num <= LR_WARMUP_STEPS:
        warm = step_num / LR_WARMUP_STEPS
        for g in optimizer.param_groups:
            g["lr"] = LR * warm
    elif step_num == LR_WARMUP_STEPS + 1:
        # Pin to base LR exactly once after warmup completes.
        for g in optimizer.param_groups:
            g["lr"] = LR
    optimizer.step()
    # Historical note: this used to round gradients to 1/8192 and Adam state
    # to 1/16384 to make cross-machine PyTorch produce identical updates. The
    # rounding rounded `exp_avg_sq` (squared gradients, typically O(1e-7)) all
    # the way to zero, which made `m / (sqrt(v) + eps)` blow up by 1e8 and
    # weights exploded after step 2. We don't need it any more: cross-machine
    # determinism now lives in the Q15 fixed-point engine (KOLMO_FIXED=1).
    # Within-machine CPU PyTorch is deterministic without intervention.


def update_history(history: list[int], new_bytes: list[int]) -> list[int]:
    """Append new bytes to the sliding-window history."""
    history = history + new_bytes
    if len(history) > CONTEXT:
        history = history[-CONTEXT:]
    return history


def append_copy_history(copy_history: bytearray, byte: int) -> None:
    """Append one byte to copy history while bounding long-file memory.

    Copy offsets are capped at COPY_WINDOW, so older bytes are never addressable
    by the compressed stream. Trim in chunks rather than every byte to avoid
    repeatedly shifting the bytearray front on long files.
    """
    copy_history.append(byte)
    if len(copy_history) > 2 * COPY_WINDOW:
        del copy_history[:-COPY_WINDOW]


def find_copy(data: bytes, pos: int, known: bytes | bytearray) -> tuple[int, int] | None:
    """Find a simple non-overlapping LZ-style match in recent known bytes.

    Returns (offset, length), where offset=1 means "copy from the previous byte".
    """
    remaining = len(data) - pos
    if remaining < COPY_MIN:
        return None

    window = known[-COPY_WINDOW:]
    key = data[pos : pos + COPY_MIN]
    best_offset = 0
    best_len = 0

    idx = window.rfind(key)
    while idx != -1:
        offset = len(window) - idx
        max_len = min(COPY_MAX, remaining, offset)
        length = COPY_MIN
        while length < max_len and window[idx + length] == data[pos + length]:
            length += 1
        if length > best_len:
            best_offset = offset
            best_len = length
        idx = window.rfind(key, 0, idx)

    if best_len < COPY_MIN:
        return None
    return best_offset, best_len


class RollingCopyMatcher:
    """Fast LZ-style matcher for the compressor.

    The old compressor called `find_copy(data, pos, copy_history)` at every
    position, and `find_copy` searched the current window with repeated
    `rfind` calls. That is fine for tiny files but too expensive for big ones.

    This matcher indexes COPY_MIN-byte keys by absolute position as soon as
    those bytes are known. At probe time it only checks positions with the same
    8-byte key, newest first, and caps the candidate chain. This is the same
    broad shape as practical LZ compressors: hash lookup first, byte compare
    only for plausible candidates.
    """

    def __init__(
        self,
        data: bytes,
        *,
        window: int = COPY_WINDOW,
        max_candidates: int = COPY_CANDIDATES,
    ) -> None:
        self.data = data
        self.window = window
        self.max_candidates = max_candidates
        self._index: defaultdict[bytes, deque[int]] = defaultdict(deque)
        self._indexed_positions: deque[tuple[int, bytes]] = deque()
        self._next_index_pos = 0

    def _index_known_prefix(self, pos: int) -> None:
        """Index every COPY_MIN-byte key fully known before `pos`."""
        limit = min(pos - COPY_MIN + 1, len(self.data) - COPY_MIN + 1)
        while self._next_index_pos < limit:
            start = self._next_index_pos
            key = self.data[start : start + COPY_MIN]
            self._index[key].append(start)
            self._indexed_positions.append((start, key))
            self._next_index_pos += 1

    def _prune_old(self, pos: int) -> None:
        min_start = pos - self.window
        while self._indexed_positions and self._indexed_positions[0][0] < min_start:
            old_start, key = self._indexed_positions.popleft()
            candidates = self._index.get(key)
            if not candidates:
                continue
            if candidates[0] == old_start:
                candidates.popleft()
            if not candidates:
                del self._index[key]

    def candidates(self, pos: int) -> list[tuple[int, int]]:
        """Return plausible (offset, length) copy candidates at `pos`.

        Candidates are newest-first (same order the matcher inspects them),
        capped by `max_candidates`, and already filtered to length >= COPY_MIN.
        The compressor can use this to choose by estimated coding cost instead
        of blindly taking the longest match.
        """
        remaining = len(self.data) - pos
        if remaining < COPY_MIN:
            return []

        self._index_known_prefix(pos)
        self._prune_old(pos)
        key = self.data[pos : pos + COPY_MIN]
        candidates = self._index.get(key)
        if not candidates:
            return []

        min_start = pos - self.window
        while candidates and candidates[0] < min_start:
            candidates.popleft()
        if not candidates:
            return []

        out: list[tuple[int, int]] = []
        checked = 0
        for start in reversed(candidates):
            offset = pos - start
            if offset <= 0 or offset > self.window:
                continue
            # Non-overlapping copy: the copied span can't read bytes that are
            # being produced by this same copy event.
            max_len = min(COPY_MAX, remaining, offset)
            length = COPY_MIN
            while (
                length < max_len
                and self.data[start + length] == self.data[pos + length]
            ):
                length += 1
            if length >= COPY_MIN:
                out.append((offset, length))
                if length == COPY_MAX:
                    break
            checked += 1
            if checked >= self.max_candidates:
                break

        return out

    def find(self, pos: int) -> tuple[int, int] | None:
        candidates = self.candidates(pos)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])
