# data/ — the member-winter cache

`member_winters.csv` is written by `load_cache(rebuild=True)` in `../lensml.py` (run from chapter 1, *The member-winter table*) and read by every chapter.
Its first line is a `#` header recording the parameters it was derived with. The committed copy says `derived by 02_applications.ipynb`, this cookbook's earlier single-notebook version, which produced it on 2026-10-20; the same derivation now lives in `lensml.py`, and a rebuild writes `derived by lensml.py`. The header records (SEASON, WINDOW, FRONT_RANGE, NINO34, n_members_per_half,
RNG_SEED, access date). One row = one member-winter: `member_id, forcing_variant, winter, nino34_son_K, prect_winter_mm_day` — the raw
SON Niño 3.4 box mean and the raw season-mean precipitation; the forced response is removed and the target defined in chapter 2,
on training members only, so the cache carries no split-dependent quantity.

| column | what it is | unit |
|---|---|---|
| `member_id` | the CESM2 LENS ensemble member the row belongs to (the form the table carries, e.g. `r10i1181p1f1`) | — |
| `forcing_variant` | the forcing half the member comes from (`cmip6` or `smbb`) | — |
| `winter` | the year label of the member-winter, inside the header's `WINDOW` | year |
| `nino34_son_K` | the raw SON Niño 3.4 box mean | K |
| `prect_winter_mm_day` | the raw season-mean precipitation over the Front Range box | mm/day |

Shape of the committed copy, each value measured from the file itself (the command in the last column, run over the lines after the `#` header with `csv.DictReader`; chapter 1 prints the member and winter counts it reads at run time):

| measurement | value | measured by |
|---|---|---|
| rows (member-winters) | 11500 | `len(rows)` |
| distinct `member_id` | 100 | `len({r["member_id"] for r in rows})` |
| forcing variants | 2 (`cmip6`, `smbb`) | `sorted({r["forcing_variant"] for r in rows})` |
| winter range | 1900–2014 | `min(int(r["winter"]) for r in rows)`, `max(...)` |
| header fields | 7 (`SEASON`, `WINDOW`, `FRONT_RANGE`, `NINO34`, `n_members_per_half`, `RNG_SEED`, `access`) | `re.findall(r"(\w+)=", first_line)` |

- **Every chapter asserts the header against `lensml.py`** (SEASON, WINDOW, FRONT_RANGE, NINO34, AUTUMN_MONTHS, and the `access=` date
  against `ACCESS_DATE`) before reading the table. A cache derived with other parameters stops the run with a message to rebuild,
  so a recorded decision (a changed `SEASON`, or a changed autumn) cannot be silently ignored. **The committed header predates the `AUTUMN_MONTHS` field**, so the file's autumn is assumed, not read: the reader takes a header
  without it as `AUTUMN_MONTHS=[9, 10, 11]` (SON, the only autumn the derivation code had when the table was written) and prints a line
  saying so. A changed `AUTUMN_MONTHS` in `lensml.py` still fails against that assumption; a rebuilt header records the field, so the
  check then reads it from the file.
- The committed copy is the **full table**: 50 members per half, every member of both halves (the header records it, and chapter 1 prints
  the member and winter counts it actually read). It was produced by a full rebuild — a background job: the one full derivation took 33 min
  on a fast link (2026-10-20); on home broadband expect an hour or more (chapter 1's `chunking` line computes the volume from the store:
  four 600-month chunks of about 133 MB, read whole, ≈ 531 MB per member-variable, three variables, ≈ 159 GB uncompressed for 100 members; the transfer is smaller) — and the
  access date was written into the header's `access=` field. **A rebuild is a new access:** set `ACCESS_DATE` in `lensml.py` to the day of the rebuild first: `load_cache(rebuild=True)` writes it into the
  header and every figure's citation; on every other read the header's `access=` is asserted equal to
  `ACCESS_DATE` in `lensml.py`, so the two cannot drift.
- A quick sample (`REBUILD = True` in chapter 1 with a smaller `N_MEMBERS_PER_HALF`) runs the same code: the test hold-out and the fold
  count then follow the data (`N_FOLDS = min(N_FOLDS_MAX, training members)`), and the chapters print the counts they actually used and say
  when the fold re-draws collapse.
- Source and required citation (NSF NCAR's Data Citation section, which the UCAR Data Terms bind users to): Grover, M., & Banihirwe, A. (2021). NCAR CESM2 Large Ensemble Cloud-Optimized Dataset. NSF NCAR CISL. doi:10.26024/y48t-q717; and Rodgers et al. 2021, doi:10.5194/esd-12-1393-2021. UCAR Data Terms of Use (CC BY 4.0).
  Access line: Community Earth System Model v2 Large Ensemble (CESM2 LENS) was accessed on 2026-10-20 from https://registry.opendata.aws/ncar-cesm2-lens. (the header's `access=` field)

## member_winters_ssp370.csv (chapter 7)

The same 100 members, boxes, DJF season and SON autumn under SSP3-7.0, winters 2016–2100 (DJF 2015 would need December 2014 from the
historical store), derived from the ssp370 stores by `derive_future_cache` in `../lensml.py` with the same `derive_member_winters`.
Its header records the experiment, window, parameters, member count and access date (2026-10-29); `load_future_cache` asserts that
the header contains `FUTURE_RECORD` (experiment, season, window, both boxes, autumn months) and that its `access=` date equals
`FUTURE_ACCESS_DATE` in `lensml.py`. `ssp370-test-protocol.md`, beside it, is the protocol chapter 7 follows, verbatim.
Chapter 7 is the only reader, under a protocol written, by the author's account, before the file was derived (self-attested; no external timestamp).
The protocol is kept exactly as written, so two of its phrases predate the published text: its title and chapter 7's printed rule say
"pre-registered", meaning fixed in advance by the author, not lodged with a registry; and its "seven limits" refer to a draft of chapter 6 that day, not in the repository history (chapter 6 now lists eight). Its second line (`Glen chose "2"`) is the author's note of choosing a fresh test; the "2" numbers an option in a list that is not in this repository. Chapter 7 compares the file's SHA-256 with its hash as committed
with the test (851d900), so any edit, even a typo fix, shows there as
"differs"; an amendment belongs in a new, dated file beside it.
