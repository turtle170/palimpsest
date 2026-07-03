# Palimpsest

A research prototype of a **self-editing scratchpad architecture**: output is
produced through iterative span-level revision rather than single-pass
autoregressive decoding. All components are trained from scratch in Rust with
[Burn](https://burn.dev) — there is no frozen pretrained base model.

## Components

1. **Drafter** (`palimpsest-drafter`) — base autoregressive model; writes an
   initial draft into a private scratchpad.
2. **Router** (`palimpsest-router`) — small model reading the Drafter's
   internal activations; outputs a per-token "attention-worthy" score to
   locate spans worth editing.
3. **Editor** (`palimpsest-editor`) — given a flagged span + context, emits a
   replacement patch.
4. **Critic** (`palimpsest-critic`) — trained on ablation-derived causal
   "weight" labels; scores which spans matter most, and provides the halting
   signal (stop editing when the marginal weight of remaining flagged spans
   drops below a threshold).

Supporting crates: `palimpsest-core` (shared types: `Span`, `Patch`,
`ScratchpadState`, `Tokenizer` trait, reusable NN blocks),
`palimpsest-data` (synthetic toy data + batching), `palimpsest-train`
(training binaries).

## Toy task

Everything is validated on a synthetic **key-value recall** task before any
real data is involved:

```
BOS  K3 V1  K0 V7  K5 V2  K6 V4  SEP  K5 QUERY  V2  EOS
     └─pair─┘                         └query┘   └answer┘
```

The model must emit the value bound to the queried key. This task has known
causal structure — ablating the queried pair must hurt the answer, ablating
other pairs must not — which gives every phase an objectively checkable
ground truth.

## Build order (phases)

Each phase produces something independently testable before the next depends
on it. **Do not wire the full loop before each component passes its
standalone checkpoint.**

| Phase | What | Checkpoint |
|-------|------|-----------|
| 0 | Workspace, toy data, tokenizer trait | `cargo build` + batch loader tests pass |
| 1 | Drafter standalone | learns toy task; save/load round-trips |
| 2 | Ablation labeling pipeline | labels sane: queried pair ≫ filler |
| 3 | Critic (supervised on ablation labels) | rank correlation ≫ random on held-out |
| 4 | Router (reads Drafter activations) | flagged tokens overlap ablation truth |
| 5 | Editor (corrupt-and-repair) | repairs reduce badness on unseen corruptions |
| 6 | Full edit loop | halts within budget; final > first draft |

## Usage

```sh
cargo test --workspace          # unit tests, CPU (ndarray) backend
cargo run --release -p palimpsest-train -- drafter   # Phase 1 training
```

Backends: `ndarray` (default, CPU) — `wgpu`/`tch` feature flags exist on
`palimpsest-train` but are unexercised so far; correctness on CPU comes
first. Core crates are generic over `B: Backend` and never name a backend.

## Explicit non-goals right now

- Real-world datasets / large-scale training
- GPU optimization
- The eventual Python port (architecture logic is kept translation-friendly,
  nothing more)
- Joint end-to-end training from random init before per-phase validation
