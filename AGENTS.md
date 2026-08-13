# Agent guide

mnema is long-term memory for agents. You do not file things under keys — you
remember prose, and newer beliefs automatically displace the older beliefs
they replace inside a fixed-size mathematical state. Use it through three
verbs and trust what they print.

## Using mnema as your memory

```console
mnema remember "<anything worth keeping>"
mnema ask "<question>"
mnema keep <hash>
```

- **remember**: prose in, nothing else. The output tells you what your write
  displaced (`displaced 8397fcd1 [attenuated 0.83] "..."`). Read it. If an
  inference is wrong — the displaced belief is NOT actually replaced by what
  you wrote — restore it immediately with `mnema keep 8397fcd1`. Hashes are
  pronouns the system prints for you; never invent one.
- **ask**: the `support ... → settled|sparse|unwritten` line comes first.
  `unwritten` means this memory holds nothing there — do not treat the listed
  neighbors as answers. `sparse` means adjacent ground exists; read hits
  critically. Hits are real remembered entries, newest truth first;
  `<<superseded>>`/`<<displaced>>` annotations mean a newer belief exists —
  prefer it.
- **Answer from the hits, in your own words.** mnema never generates prose;
  you are the reader. Cite `at` timestamps when freshness matters.

## Asking well

Resolution takes the best single road into a belief — commit to one road
cleanly:

- **Short and natural** ("can commits jump the queue?") rides stored
  question-aliases; **detailed and specific** rides the content match. The
  worst query is the middle-length muddle that commits to neither.
- **Specificity ≠ verbosity.** Detail that discriminates (anchor terms, the
  exact aspect) aligns; detail that narrates (why you ask, your situation)
  pools noise into the query vector. Say only what selects.
- **One belief per question.** Bundled questions land between their answers.
  Ask twice.
- **Assert your hypothesis** when you suspect what the memory says —
  claim-to-claim matching is the strongest geometry.
- On `sparse`, rephrase along an axis: more specific (toward content) or more
  plain (toward aliases) — not just differently.

Add `questions` to a memory only when you know phrasings its text lacks —
measured: question-aliases help phrasing-gap corpora, and cost a little on
curated doc corpora whose misses are sibling co-answers.

## One vault or many?

The rule: **things that should supersede each other belong in ONE vault;
things that should be compared belong in SEPARATE vaults.** Supersession only
acts within a store — across vaults, disagreeing memories surface side by
side, tagged by origin, which is what you want across authorities.

- **A vault is the unit of sharing.** You serve a whole store or none of it —
  keep personal memory separate from anything a teammate might mount.
- **A vault is the unit of authority.** Doctrine, operational logs, and
  scratch notes shouldn't erode each other by mere semantic proximity — give
  each its own store.
- **A vault is the unit of lifecycle.** Derived stores (rebuilt from git/docs
  by re-running ingest) are disposable; authored stores (`remember`) are
  precious logs. Never mix them in one append-only file.
- **Topic prefixes are categories WITHIN one authority** — e.g. all of a
  repo's markdown in one vault, with `path#heading` topics as the category
  system and internal supersession fully legitimate.
- Splitting costs nothing at read time: `ask` fans across local + all vaults
  and interleaves by score. The read side reunifies what the write side
  separates.

## Writing memories well

Relevance is decided at write time, not query time — the matrix returns what
was folded. Discipline for every `remember`:

1. **One belief per entry.** The unit of memory is the unit of retrieval; an
   omnibus note matches everything weakly. Split multi-topic updates into
   separate remembers.
2. **Distill, don't dump.** Remember the conclusion ("we chose X over Y
   because Z"), never the transcript that produced it. Distilled beliefs also
   make displacement inference sharp.
3. **Topic anything with a lifecycle.** Will a future write *replace* this
   (a policy, an owner, a schedule, a status)? → `--topic name` for exact
   supersession. One-off facts and observations stay keyless.
4. **Scope stores by register.** Personal preferences, project rulings, and
   chat digests belong in separate stores (`MNEMA_STORE` per context) —
   mixed-genre corpora blur verdicts and misfire displacement. Recombine at
   read time with vaults or `merge`.
5. **Curate immediately.** Read the `displaced` lines after each remember and
   `keep` wrong inferences while you have context. `forget` noise, test
   entries, and accidents — both commands are exact and cheap.
6. **Fix the corpus, not the phrasing.** If questions keep landing sparse,
   the store is missing a layer (e.g. definitional docs vs. operational
   decisions) — ingest the missing source instead of torturing queries.
7. **History before life.** Bulk-ingest archives before daily writes (the
   fold runs in log order), and check `mnema stats` occasionally — act on
   headroom warnings before recall degrades.

## Speed

If `ask`/`remember` feel slow (~10s), start the warm-model daemon once:
`mnema serve &` — every subsequent command is ~1s. The CLI finds it
automatically and falls back gracefully if it dies.

## Which memory you're talking to

Resolution order: `--store PATH` flag > `MNEMA_STORE` env var > `~/.mnema`.
Keep separate stores per scope (personal, project, team). Create one with
`mnema init --store <path>`.

## Vaults: other people's memories

`mnema vault list` shows every memory you can read: your local store plus
named read-only vaults (a colleague's, a team's) mirrored from URLs or paths.

```console
mnema vault add https://<host>/mnema --name alice
mnema ask "did anyone decide X?"            # local + all vaults, hits tagged (vault: alice)
mnema ask --from alice "..."                # one vault only
mnema ask --local "..."                     # your memory only
```

Writes NEVER go to a vault — `remember` is local by construction. To share
your memory, serve your store directory over HTTP (see `src/mnema/vaults.py`).

Phrase vault questions in the THIRD person ("who owns this store", "what did
alice decide about X") — a vault's memories speak in their owner's voice, so
"my"/"I" in your question refers to the wrong person. Support verdicts are
calibrated conservatively: on small or conversational vaults, a correct answer
may sit in the `sparse` band — read hits critically rather than discarding
them. Hit scores are retrieval strengths, not the same scale as support.

## Working on this repo

- `uv sync --extra dev` then `uv run pytest` — the suite runs in <1s, no
  model download needed (embedding is faked in tests).
- The tests assert *algebraic laws*, not implementation details. Any change
  to fold semantics (core.py, the ops builder in store.py) must come with a
  law test, and must keep `test_stale_never_wins_within_capacity` and
  `test_keep_revokes_displacement_exactly` green — those two ARE the product.
- `log.jsonl` is append-only and authoritative; vectors and state are
  disposable caches. Never write code that edits or reorders log lines.
- Store `config.json` is immutable after init. Changing model/dim/seed means
  a new store.
- Real-model smoke test: `uv run mnema --store /tmp/m init && uv run mnema
  --store /tmp/m remember "hello" && uv run mnema --store /tmp/m ask "hello"`
  (first run downloads ~1.3 GB and needs network).
