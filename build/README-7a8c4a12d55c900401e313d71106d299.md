# data/ — the Module 2 member-winter cache

`member_winters.csv` is written by `02_applications.ipynb` (cell "Deriving the member-winter table") and read by every later cell.
Its first line is a `#` header recording the parameters it was derived with (SEASON, WINDOW, FRONT_RANGE, NINO34, n_members_per_half,
RNG_SEED, access date). One row = one member-winter: `member_id, forcing_variant, winter, nino34_son_K, prect_winter_mm_day` — the raw
SON Niño 3.4 box mean and the raw season-mean precipitation; the forced response is removed and the target defined in the notebook (2b),
on training members only, so the cache carries no split-dependent quantity.

| column | what it is | unit |
|---|---|---|
| `member_id` | the CESM2 LENS ensemble member the row belongs to (the form the table carries, e.g. `r10i1181p1f1`) | — |
| `forcing_variant` | the forcing half the member comes from (`cmip6` or `smbb`) | — |
| `winter` | the year label of the member-winter, inside the header's `WINDOW` | year |
| `nino34_son_K` | the raw SON Niño 3.4 box mean | K |
| `prect_winter_mm_day` | the raw season-mean precipitation over the Front Range box | mm/day |

Shape of the committed copy, each value measured from the file itself (the command in the last column, run over the lines after the `#` header with `csv.DictReader`; 2a prints the member and winter counts it reads at run time):

| measurement | value | measured by |
|---|---|---|
| rows (member-winters) | 11500 | `len(rows)` |
| distinct `member_id` | 100 | `len({r["member_id"] for r in rows})` |
| forcing variants | 2 (`cmip6`, `smbb`) | `sorted({r["forcing_variant"] for r in rows})` |
| winter range | 1900–2014 | `min(int(r["winter"]) for r in rows)`, `max(...)` |
| header fields | 7 (`SEASON`, `WINDOW`, `FRONT_RANGE`, `NINO34`, `n_members_per_half`, `RNG_SEED`, `access`) | `re.findall(r"(\w+)=", first_line)` |

- **The notebook asserts the header against its parameter cell** (SEASON, WINDOW, FRONT_RANGE, NINO34, AUTUMN_MONTHS, and the `access=` date
  against `ACCESS_DATE`) before reading the table. A cache derived with other parameters stops the run with a message to set `REBUILD = True`,
  so a recorded decision (Exercise 1's `SEASON`, or a changed autumn) cannot be silently ignored. The committed header predates the
  `AUTUMN_MONTHS` field: the notebook reads a header without it as `AUTUMN_MONTHS=[9, 10, 11]`, the only autumn
  the derivation code had when the table was written, so a changed `AUTUMN_MONTHS` still fails against it; a rebuilt header carries the field.
- The committed copy is the **full table**: `N_MEMBERS_PER_HALF = 50`, every member of both halves (the header records it, and 2a prints
  the member and winter counts it actually read). It was produced by setting `REBUILD = True` in the notebook — a background job: the course's one full derivation took 33 min
  on a fast link (2026-10-20); on home broadband expect an hour or more (the notebook's 2a `chunking` line computes it from the store: four 600-month chunks of about 133 MB, read whole, ≈ 530 MB per member-variable, three variables, ≈ 160 GB for 100 members) — and the notebook wrote the access date into the header's `access=` field. **A rebuild is a new access:** the `REBUILD = True` branch stamps the
  day it reads S3 into `ACCESS_DATE`, the header and every figure's citation; on every other run the notebook asserts
  the header's `access=` equals the parameter cell's `ACCESS_DATE`, so the two cannot drift.
- A quick sample (`REBUILD = True` with a smaller `N_MEMBERS_PER_HALF`) runs the same code: the test hold-out and the fold count then follow
  the data (`N_FOLDS = min(N_FOLDS_MAX, training members)`), and the notebook prints the counts it actually used and says when the fold
  re-draws collapse.
- Source and required citation (NSF NCAR's Data Citation section, which the UCAR Data Terms bind users to): Grover, M., & Banihirwe, A. (2021). NCAR CESM2 Large Ensemble Cloud-Optimized Dataset. NSF NCAR CISL. https://doi.org/10.26024/y48t-q717; and Rodgers et al. 2021, https://doi.org/10.5194/esd-12-1393-2021. UCAR Data Terms of Use (CC BY 4.0).
  Access line: Community Earth System Model v2 Large Ensemble (CESM2 LENS) was accessed on 2026-10-20 from https://registry.opendata.aws/ncar-cesm2-lens. (the header's `access=` field)
