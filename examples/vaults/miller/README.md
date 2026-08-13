# The Miller vault

The object-capability / agoric canon — Mark Miller's research line from the
1988 agoric-systems papers through *Robust Composition* (2006) and the
distributed-electronic-rights work (2013). Eight text papers plus three
optional OCR'd scans, each ingested with `--at` set to its publication date,
so the fold's time axis follows the intellectual history rather than file
mtimes.

```console
$ bash examples/vaults/miller/build.sh          # or: build.sh /path/to/store
```

The build fetches each paper from a public mirror (agoric.com's paper archive,
the Internet Archive, USENIX, a HP Labs mirror — see `manifest.tsv`) into the
store's own `sources/` directory and ingests from there. No paper text ships
in this repo: the papers are their authors' copyrighted works, and a derived
store is rebuilt from sources by re-running ingest anyway — the manifest and
builder ARE the vault, in reproducible form.

The three 1988 papers are image-only scans with no text layer, so they ride
an optional OCR tier: if `tesseract` and `pdftoppm` (poppler) are on PATH the
build rasterizes and OCRs them; otherwise it skips them with a note and the
vault is still complete for the eight text papers. Old TeX PDFs also carry the
broken-ligature caveat documented in `AGENTS.md` ("Writing memories well",
point 9) — dense matching tolerates it, exact-term search does not.

```console
$ bash examples/vaults/miller/build.sh
$ mnema vault add ~/.mnema-miller --name miller
$ mnema ask "what do capabilities solve that access control lists cannot?"
settled — a strong match exists; the top result is trustworthy
<hit h="..." at="2003-03-01" kind="doc" vault="miller" topic="sources/capability-myths-demolished.pdf#...">
  ...
  <related h="..." vault="..." cos="...">...</related>
</hit>
```

Once mounted, the vault joins every ask — and its hits take the relate hop
into your other stores, so a question about your own architecture can surface
the paper that named the pattern.
