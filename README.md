# mnema

**Memory for agents that works like memory** — you don't file things under keys,
you just *remember*. Everything remembered folds into one fixed-size matrix
where **newer beliefs physically subtract the older beliefs they replace**.
Stale knowledge doesn't get down-ranked, filtered, or garbage-collected — it is
structurally absent from the state, by arithmetic.

```console
$ mnema init
$ mnema remember "the deploy freeze is on: no production deploys on fridays"
$ mnema remember "deploy freeze lifted — continuous deploys all week, canary-gated"
  displaced 8397fcd1  [attenuated 0.83]  "the deploy freeze is on: no production deploys..."

$ mnema ask "can we deploy on friday?"
settled — a strong match exists; the top result is trustworthy
navigate: mnema show <hash> opens any memory in full · mnema ask --from <vault> "..." scopes a vault · refine the question to keep moving

<hit h="2d2c364f" at="2026-08-13" kind="note" topic="ops/deploy-freeze">
  deploy freeze lifted — continuous deploys all week, canary-gated
</hit>

$ mnema ask "what is the office wifi password?"
unwritten — nothing matches this well; nearest content shown, verify before trusting
```

No key was ever given. The second memory *found* the first one — because it
landed on it in meaning space — and attenuated it. Ask about deploys and you
get the current truth; ask about something never remembered and the memory
**says so** instead of guessing.

## Install

Requires Python 3.11/3.12. The embedding model (~1.3 GB, downloaded once) runs
locally — Apple Silicon GPU, CUDA, or CPU. Nothing ever leaves your machine.

```console
$ git clone <this repo> && cd mnema
$ uv sync            # or: pip install -e .
$ uv run mnema init  # creates ~/.mnema
```

## How agents use it

The intended consumer is an agent with a shell. The contract is three verbs
and a pronoun:

```console
mnema remember "<anything worth keeping>"     # write. no key, no schema.
mnema ask "<question>"                        # read. verdict + sources.
mnema show <hash>                             # dereference any printed hash.
mnema keep <hash>                             # veto a wrong displacement.
```

The loop that makes it work as long-term memory:

1. **Remember freely.** Prose in, nothing else. The system infers what each new
   memory displaces (floor-gated similarity, weight capped below 1) and prints
   what it did — the agent sees `displaced 8397fcd1 [0.83] "..."` and can
   immediately `keep 8397fcd1` if the inference was wrong. Hashes are pronouns
   the system hands you, never names you must invent — and every printed one
   dereferences to its full memory with `mnema show <hash>`.
2. **Trust the verdict.** `ask` returns `settled` / `sparse` / `unwritten`
   support before any results. `unwritten` means *this memory holds nothing
   there* — the difference between an assistant that says "I don't know" and
   one that confabulates.
3. **Answer from the sources.** `ask` returns real remembered entries verbatim
   (with timestamps, hashes, and displacement annotations), not generated
   summaries. The agent does its own reading; the log stays the authority.

Scaling, stated honestly. The **fold update** is O(1) in history; today's
write *bookkeeping* (dedupe, topic lookup, displacement inference) scans the
log and vectors — linear with small constants, fixable with a persistent
index if write volume ever demands it. The **read side** is linear in
history with no quadratic terms — full K/V scans plus currency evaluated
only on entries in a recorded supersession relationship. Practical limits
depend on D and on the fraction of memories participating in supersession,
and need benchmarking at scale rather than assertion. The architecture's
scale path is candidate-set retrieval — ANN/lexical prefilter, expand by
supersession-family edges, evaluate fold-currency inside that subgraph —
which changes none of the fold's semantics.

## Pointing at different memories

A memory is a directory. Which one you're talking to is resolved as:

```console
mnema --store ./project-mem ...       # explicit flag, highest priority
MNEMA_STORE=~/mem/org mnema ...       # environment variable
mnema ...                             # default: ~/.mnema
```

So an agent can keep **separate memories per scope** — personal, per-project,
per-team — by exporting `MNEMA_STORE` in the relevant context, and combine
them when needed:

```console
$ mnema --store ~/mem/team-a merge ~/mem/team-b --out ~/mem/org
```

Merge interleaves the two logs by timestamp, reuses both stores' vectors (no
re-embedding — the expensive part never re-runs), and refolds once (pure
matrix math, seconds). The result is exactly the state a single store would
hold had both histories been written into it from the start — supersession
acts *across* origins, one memory with no seams. Same-config stores only:
matching model/seed/dimension is what makes the two stores share one meaning
space, so there is nothing to translate. A useful way to hold it: embeddings
are *facts* (established once, never recomputed); the fold is an *opinion*
(which beliefs survive the whole ordered history) — and merge is precisely a
re-derivation of the opinion over the union of facts.

## The identity ladder

Three levels of declared identity, one dimension: how much erasure-authority
one write holds over another.

| level | identity means | supersession | use for |
|---|---|---|---|
| bare prose | "similar belief" | graded by meaning, inferred, revocable | observations that drift |
| topic | "same slot" | total, by exact string equality | beliefs that version |
| store | "same authority" | none — disagreement surfaces, tagged | different believers |

Two mechanisms, two triggers: **equality is the law** — an identical topic
string fully replaces, regardless of content similarity (that is what
declaring a topic means; `remember` warns when a declared replacement is
semantically unrelated, the signature of a sibling mistaken for a version).
**Similarity is the physics** — everything erodes neighbors in proportion to
address closeness, including similar topic strings, since topics are embedded
text. Prefixes (`linting/tsgo`, `linting/tslint`) have zero mechanics — the
machine never parses a topic — they organize for humans and cluster mildly in
embedding space. A topic is a slot, not a folder: before reusing one, the
question is never "is this related?" but "does this replace what's there?"
Siblings get sibling topics; category questions aggregate them through
content matching automatically.

## Vaults: read many, write your own

A vault is someone else's memory at a known address — a teammate's store on
your tailnet, a team store on a LAN — that joins your queries read-only:

```console
$ mnema vault add https://alice-mbp.tail1234.ts.net/mnema --name alice
$ mnema vault list
local            ~/.mnema                                    412 memories
alice            https://alice-mbp.tail1234.ts.net/mnema     876 memories

$ mnema topics --from alice                         # just the topic slots, no content
alice: 214 topics (876 topical writes, 12 anonymous)
  ops/deploy-freeze    2026-08-13  6fe00a06  (2 writes)
  ops/retro            2026-08-01  90635959

$ mnema ask "did anyone rule on retry semantics?"   # local + every vault
<hit h="8397fcd1" at="2026-08-11" kind="note" vault="alice">
  retries are per-event with capped backoff, ruled at ...
</hit>

$ mnema ask --from alice "..."                      # one vault, by name
$ mnema ask --local "..."                           # your memory only
$ mnema remember "..."                              # ALWAYS local — writes
                                                    # cannot reach a vault
```

When a question spans stores, the top hit also takes one relate hop: its own
value vector probes every *other* consulted store (never its origin), and
`<related h="8397fcd1" vault="alice" cos="0.75">` blocks inside the first hit
surface beliefs that connect across vaults. The `cos` is a relatedness
weight, not a support score — claim-to-claim cosines run hot, and these
blocks never carry verdict semantics. Follow one with `mnema show <hash>` (the full memory,
whichever vault it lives in) or `mnema ask --from <vault>` when the
connection matters.

The hop needs no setup beyond mounting: every vault already joins every ask,
the ranking runs over the union of all consulted stores, and the probe is
the top hit's own stored vector rather than your question. So one query
shows two things — how your *question* resolves across everything, and how
your *answer* relates to everything else; the second is the browsing system.
A stored memory is a distilled, specific claim, and claim-to-claim matching
is the strongest retrieval geometry, so detailed one-belief-per-entry
memories pay twice: every well-written memory is already an ideal query. The
loop an agent runs: ask, read the answer, `mnema show` a related hash,
`ask --from` that vault if it matters. One level deep, similarity only,
never a verdict — and a colleague's vault participates with zero
coordination: your top hit surfaces whatever they hold nearest to it.

Serving your store is one command, because its files are append-only and its
config immutable:

```console
$ tailscale serve --bg --set-path /mnema ~/.mnema   # tailnet (HTTPS, private)
$ cd ~/.mnema && python -m http.server 8377         # LAN
```

Vault addresses resolve through a **source interface** (`mnema.vaults.SOURCES`):
http/https/file URLs and bare paths ship built-in, and an adapter for S3, git,
ssh, IPFS — whatever — is a ~5-line class with one `fetch(filename) -> bytes`
method, registered by URL scheme. Vaults must share the local store's config
(model, dimension, seed) to be comparable; mismatches are skipped with a
warning, never silently mixed. Vaults are not transitive — a vault's own
vaults are ignored.

### Example vaults

`examples/vaults/miller/` builds a starter vault from the object-capability /
agoric research canon: a manifest of public paper URLs plus a one-command
builder that fetches and ingests them dated by publication (no paper text
ships in the repo — derived stores are rebuilt from sources). Once built it is
a store like any other: mount it locally with `mnema vault add`, or share it
with teammates by serving the directory exactly as you would your own.

## How good is it?

Two layers, two answers. The **storage/currency layer** (delta fold read at
exact addresses) benchmarks at ~0.97 recall@1 with zero stale-wins within
capacity. The **question-resolution layer** (your English → an address) was
measured end-to-end on a hard probe set — 20 paraphrased questions against a
narrow 42-entry doctrine corpus — and the finding is that **the verdict
predicts its own accuracy**:

| verdict | share of probes | top-1 | top-3 |
|---|---|---|---|
| settled | 70% | **100%** | 100% |
| sparse | 30% | 67% | 100% |

Read it as a contract: `settled` answers are trustworthy; `sparse` means the
right entry is probably in the list but read critically; `unwritten` means
stop. Every hard miss in the benchmark was pre-announced by its own verdict.
Queries are always plain language — addresses are resolved internally and
never user-supplied.

The factored composition was established by ablation across four probe
classes (paraphrase resolution on two corpora, cross-vault polysemy,
supersession pairs): the full product wins or ties every class that its
factors have authority over — currency turns the supersession pairs from
1/2 to 2/2, the gated lexical view turns polysemy from 3/6 to 5/6 and a
1,248-entry corpus from 11 to 13/16 — and each factor is near-inert outside
its jurisdiction. `benchmarks/` holds the harness; probes are corpus-local
and never committed. Caveat these numbers honestly: the probe sets are
small and corpus-specific — treat the verdict thresholds as per-corpus
calibration targets, not universal constants.

## Performance

CLI latency is dominated by loading model weights into a fresh process
(~seconds). Run the warm-model daemon and every command drops to ~1s:

```console
$ mnema serve &        # one per model; every store on that model benefits
```

The CLI detects a running daemon automatically (unix socket, per-model) and
falls back to in-process loading when there isn't one — no flags, no config.
Commands that never touch the model (`log`, `vault list`, `keep`) are fast
regardless.

## What's in a store

```
config.json   model, dimension D, seed, β, σ — immutable after init
log.jsonl     append-only memories — THE source of truth
vec_v.f16     value vectors (write-once, one row per memory)
vec_k.f16     key vectors
state.npz     the matrix S, the support matrix Λ, and the fold cursor
views.npz     read-side views: inverted lexical index, topic/supersession maps
```

The log is authoritative; everything else is a disposable cache. Delete
`state.npz` and the next command refolds it. Every command starts with
*catch-up* (embed and fold whatever the log has that the state hasn't), so
concurrent writers and cross-session use need no coordination.

## Bulk import

A translator is any function `path -> [entries]`. Built-ins:

```console
$ mnema ingest --format jsonl facts.jsonl        # {"text": ..., "at"?: ..., "topic"?: ...}
$ mnema ingest --format ledger "logs/**/*.jsonl" # tagged event ledgers
$ mnema ingest --format paper --at 2006-05-15T00:00:00Z thesis.pdf
```

`paper` ingests whole documents: non-markdown formats (PDF, DOCX, EPUB, ...)
convert through the optional anydoc converter — `pip install "mnema[paper]"`
if it's missing — then sections split at headings and paragraphs pack into
~300-word chunks so every chunk fits the 350-word dense-index horizon
(measured on a 207-page dissertation: heading-only sectioning left 45% of
sections past the horizon; packing leaves 9.7%, all single unbreakable
paragraphs). Chunk topics extend the section slot (`path#slug`,
`path#slug/2`, ...) — disjoint addresses within one paper, stable across
re-ingests — and pure-navigation sections (contents, lists of tables/figures,
index) are dropped. `--at` stamps one ISO timestamp on every entry: archival
documents should carry their publication date, not the file's mtime.

**Or skip Python translators entirely and pipe:** `ingest` reads JSONL from
stdin with `-`, so a translator is any shell pipeline that emits
`{"text": ..., "at"?: ...}` lines:

```console
$ jq -c '{text: .message, at: .ts}' slack-export.json | mnema ingest --format jsonl -
$ git log --format='{"text": %s, "at": "%aI"}' | mnema ingest --format jsonl -
$ ./my-scraper | jq -c '...' | mnema ingest --format jsonl -
```

Slack exports, git histories of markdown docs, decision logs, fact streams —
if you can emit `{text, at}` lines from anything, mnema folds it. Python
translators (`src/mnema/translate.py`, ~15 lines) exist for formats you ingest
repeatedly.

## The mathematics

### Write time: the fold

Every derived view — the state `S`, the density `Λ`, the vectors, the
supersession graph — is a fold of the same log, and they advance together as
**one fused product fold**: `catch_up` is the single incremental transition
`FoldState_{n+1} = step(FoldState_n, e_{n+1})`, continued from its
checkpoint, never recomputed per view. The rule: anything derivable
incrementally from the ledger advances in the same traversal.

Every memory embeds to a unit content vector `v`; its address `k` is the
embedding of its topic (or `v` itself when keyless), lifted through a fixed
seeded random projection to dimension `D` (Johnson–Lindenstrauss preserves
inner products, and `D` decouples capacity from the embedder). Memories fold
in timestamp order into one `D×D` matrix by the delta rule:

```
S ← S(I − βkkᵀ) + βvkᵀ          # read belief at k, SUBTRACT it, write new
Λ ← Λ + kkᵀ                     # address density (capacity gauge)
```

With orthonormal addresses this is *exactly* last-write-wins (zero residue —
a law test); with real quasi-orthogonal addresses the residue is O(1/√D).
Keyless writes additionally attenuate the few prior memories they land
nearest (weight < 1 ⇒ the operator is invertible ⇒ every inference is
revocable), and the inference is itself a log event. Capacity ≈ one *live*
address per dimension: superseded history refunds its space.

The log is the only truth. `S`, `Λ`, and the vectors are derived caches —
`forget` and `--as-of` work by *re-deriving* from a corrected or truncated
event set, which is why exact undo needs no invertible operators.

### Read time: a product of scoped authorities

Retrieval is a factored score. Each factor answers ONE question, holds
authority over only that question, and carries a jurisdiction gate — the
principle of least authority, applied to relevance:

```
resolution   rᵢ = max( ⟨q,kᵢ⟩ , ⟨q,vᵢ⟩ , maxⱼ⟨q,aᵢⱼ⟩ )   is it about this?
             — dense cosine against address, content ("the answer's own
               embedding is a key"), and the entry's optional question
               aliases aᵢⱼ, which fold into their parent. max, not sum:
               these are alternative DOORS to one belief, not independent
               evidence — one open door suffices, and adding an address can
               never hurt its own entry. Authority: paraphrase & meaning

lexical      BM25(q, entryᵢ) · g               does it contain the rare term?
             g = clip((maxIDF(q) − 3)/3, 0, 1) — the gate: this view votes
             ONLY when the question holds a corpus-rare term; a zero BM25
             score is no evidence, never a rank; authority: jargon, coined
             vocabulary, polysemy resolution

fusion       RRF over ONE candidate pool       reciprocal-rank fusion of the
             two views — errors of dense (blurs jargon) and lexical (blind
             to paraphrase) are near-disjoint, so agreement multiplies
             evidence. A zero lexical score contributes NO rank credit
             (absence of evidence is not a rank)

currency     cᵢ = ⟨S·kᵢ, vᵢ⟩ / max over the    does the belief still stand?
             entry's DECLARED supersession      — reads the write-time
             family (same topic, or a live      subtraction's testimony,
             displacement edge)                 normalized only among recorded
             rivals. Unrelated entries hold currency 1 by definition —
             semantic similarity finds things; recorded history decides what
             supersedes what. Authority: supersession

score        = RRF(r, lexical; g) × c
support      = max rᵢ  →  settled / sparse / unwritten
```

The raw vectors are never subtracted (they are the re-derivable facts), so
the ranking must consult `S` for currency explicitly — the currency factor
is not recomputation, it is the read-time *interface* to the write-time
subtraction.

### Cross-pool semantics: how vaults compose

A query may span the local store and many vaults. The composition rules:

1. **Cosines share one scale; ranks do not.** Dense resolutions concatenate
   across stores directly (config equality — model, dim, seed — is enforced,
   so all stores inhabit one geometry). Rank-based fusion is only meaningful
   within one candidate pool, so BM25 statistics, IDF rarity, and RRF are
   computed once over the **union** of all consulted stores — a term's
   rarity is judged against everything you can see.
2. **Currency is store-local.** Supersession is a fact about one store's own
   history; one vault's writes never erode another's. Across vaults,
   disagreeing beliefs surface side by side, tagged by origin — supersede
   within, compare across.
3. **Support is the global max** of resolution, so the verdict reflects the
   best address anywhere in the constellation.

Displayed hit scores are resolution cosines (a meaningful scale); ordering is
by the full composition.

## Properties

| Property | Mechanism |
|---|---|
| No keys, no schema | prose is its own address; displacement is inferred, reported, and vetoable |
| Append: fold update O(1) in history | one rank-one update; bookkeeping scans are linear today (index-fixable); no reindex, no compaction |
| Constant-size state | `D×D` floats at 1 memory or 1M memories |
| Stale beliefs can't compete | subtraction at write time (measured 0/75 stale-wins on a real 867-entry corpus at D≥2048) |
| Knows what it doesn't know | nearest-address support: settled / sparse / unwritten |
| Inference is safe | displacement weights capped below 1 — invertible, logged, `keep`-revocable |
| Time travel | `ask --as-of <ISO>` answers from the state as of any moment |
| Merge without re-embedding | interleave logs, reuse every vector, refold once (pure GEMMs, seconds) — exact by associativity |
| Capacity self-awareness | `mnema stats` estimates live beliefs vs D and warns before quality degrades |
| Local and private | embeddings computed on-device; nothing leaves the machine |

**Capacity:** ~one *live* belief per dimension (default D=4096), and only live
beliefs count — displaced and superseded history refunds its space. Raise
`--dim` at init or run multiple stores if your live surface is bigger.

## Advanced

- `--topic <name>` on remember: declared addresses with *exact* last-write-wins
  supersession — for automation that owns a stable vocabulary (deploy bots,
  config state). Human and agent prose doesn't need it.
- `--slot name=value`: bind role/value structure into an address
  (`--slot team=platform`), queryable with the same slots.
- `init --sigma <s>`: route addresses through random Fourier features — a
  locality dial for how far displacement and generalization reach.
- `init --model <name>`: any sentence-transformers model; see
  `src/mnema/embed.py` for the vetted registry.
- `init --beta <b>`: β=1 (default) makes declared supersession a true erasure;
  β<1 makes *all* forgetting attenuation.

## What it deliberately is not

- **Not generative.** It routes to real memories; it never writes prose.
- **Not enumeration.** The state answers similarity-shaped questions; "list
  everything" reads the log.
- **Not a vector database — a temporal/supersession layer that could sit
  behind any retrieval mechanism.** Reads currently scan stored vectors
  (linearly, index-free), but the vector store is just the address resolver:
  the log is history, and the fold is a compact representation of which
  beliefs currently survive it — supersession as arithmetic, currency, time
  travel, exact undo, associative merge. Swap the resolver (ANN, BM25,
  anything); the fold's semantics ride along unchanged.

## The laws are tested

`tests/` asserts the algebra on random data, no model required: chunked fold ≡
sequential; orthonormal addresses give *exact* last-write-wins; stale never
wins within capacity; segment associativity; support separates written from
unwritten ground; displacement attenuates measurably and `keep` restores the
control state to float precision; cross-session catch-up matches one-shot
folds bit-for-bit.

```console
$ uv run pytest
```

## Background

The update rule is the delta rule (Widrow–Hoff, 1960) as it appears in
fast-weight and linear-attention architectures (Schlag et al. 2021; DeltaNet —
Yang et al. 2024, whose chunked UT-transform this implements). The observation
here is that an append-only memory log gives the rule what it never has in
neural networks: *exact, auditable addresses to overwrite* — which turns
"approximately forgets" into measurable, revocable, zero-staleness memory.
