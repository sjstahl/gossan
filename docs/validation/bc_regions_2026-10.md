# Validation on larger BC areas (October 2026)

`scripts/validate_regions.py` ran Gossan end to end on four British Columbia areas of 7,000 to 16,000 km²
(two to four times the original East Kootenay test area): harvest of every default BC source, the
prospectivity model with 5-fold spatial cross-validation (2 km gap, blocks of about 20 km), the same with
`occurrence_class: metallic`, the knowledge-driven score, and every analysis step. The raw numbers are in
the `validation_*.json` files beside this note.

## Results (default settings, all occurrences)

| Region | km² | Cells (100 m) | Occurrences (metallic) | Forest AUC | Forest effort-matched AUC | Forest top-5% capture | Knowledge AUC | Knowledge effort-matched AUC |
|---|---|---|---|---|---|---|---|---|
| Boundary (Greenwood, Rossland) | 7,349 | 734,882 | 496 (375) | 0.83 ± 0.03 | 0.65 ± 0.07 | 38% | 0.75 | 0.60 |
| East Kootenay, wide | 16,291 | 1,629,053 | 483 (380) | 0.66 ± 0.10 | 0.55 | 16% | 0.66 | 0.57 |
| Quesnel Trough, Cariboo | 7,818 | 781,832 | 418 (356) | 0.65 ± 0.10 | 0.59 | 14% | 0.67 | 0.61 |
| Stewart (Golden Triangle) | 7,468 | 746,810 | 741 (710) | 0.74 ± 0.04 | 0.62 | 11% | 0.77 | 0.63 |

Chance is 0.5 for both AUCs and 5% for the capture. "Knowledge" is the rule-based score, which fits nothing,
checked on the same held-out blocks.

With `occurrence_class: metallic` the forest's effort-matched AUC was 0.71, 0.59, 0.65 and 0.61 (plain AUC 0.87,
0.74, 0.73, 0.74), against 0.65, 0.55, 0.59 and 0.62 with all occurrences.

Knowledge layers, each alone against all occurrences (AUC): faults 0.69 / 0.65 / 0.63 / 0.71; unit contacts
0.73 / 0.55 / 0.57 / 0.70; intrusive rocks **0.35** / 0.51 / 0.49 / 0.69; stream-sediment pathfinders 0.75 / 0.60 /
0.64 / 0.66 (same region order).

## Running cost

| Region | Harvest | Two model runs with CV | Analysis steps |
|---|---|---|---|
| Boundary, 0.73 M cells | 67 s | 386 s | 49 s |
| East Kootenay wide, 1.63 M cells | 122 s | 1,184 s | 148 s |
| Quesnel, 0.78 M cells | 117 s | 466 s | 63 s |
| Stewart, 0.75 M cells | 61 s | 476 s | 53 s |

Peak memory of the process was 2.2 GB, reached at 1.63 M cells (the default `max_cells` of 2 M, about 20,000 km²
at 100 m, is therefore workable on 8 GB or more). Model time grows faster than area: 2.2 times the cells took
about 3 times as long, because every fold refits all the bagged forests. The harvest returned every layer in
every region with no failures. A real run of the combined leads, the analysis report and `gossan deliver` on
Boundary worked; the client package there is 4.6 MB of map plus 1.8 MB of GeoPackage and three rasters.

## What it showed

1. **The forest has real but modest skill that varies by region.** Effort-matched AUC 0.55 to 0.65, top-5%
   capture 11% to 38% where chance is 5%. Boundary is best; East Kootenay wide, which contains the original
   test area, is weakest, as it was there (0.55).
2. **The unfitted knowledge score is as good as the forest in three of four regions** on the same held-out blocks
   (plain AUC 0.66 vs 0.66, 0.67 vs 0.65, 0.77 vs 0.74; effort-matched it is equal or ahead in three), and behind
   only at Boundary (0.75 vs 0.83), where the learned unit favourability (the forest's top feature) is strong. The
   fold-to-fold spread is 0.04 to 0.10, so differences of a few hundredths are noise; the finding is that
   learning from the occurrences adds little over simple geological rules in most of these areas.
3. **A rule that works in one terrane fails in another.** "Near intrusive rocks" is contradicted at Boundary
   (AUC 0.35: occurrences sit away from the mapped intrusions) and informative at Stewart (0.69). The per-layer
   AUC table in the report is what shows this; the generic weights are not right everywhere, which is why the
   deposit-model presets and weight overrides exist.
4. **Metallic-only labels helped in three of four regions** (effort-matched +0.06, +0.04, +0.06; Stewart is
   already 96% metallic and unchanged). `occurrence_class` is still "all" by default; this is evidence for
   recommending "metallic" to BC clients, not a change made.
5. **One bug found and fixed:** at Stewart the decision-table step crashed ("Encountered all NA values") when
   every sample near a target had no usable pathfinder index. It now leaves those columns empty (regression
   test in `tests/test_analyze.py`).

## Limits of this validation

- One run per region with one seed, and 5 folds: the numbers carry the spread shown, and a ±0.05 difference
  between two figures is not a finding.
- Occurrences are the only labels, and they cluster where people looked. Effort-matched AUC corrects for this
  only partly (ARIS and tenure density as proxies for effort); none of these figures measures finding new
  deposits, which needs a prospective test.
- The regions were chosen by hand for different terranes; four is a small sample.
- Not exercised on real data: magnetics (no grid obtained), ASTER (no scenes downloaded), digitized drill
  results (no approved tables). No cross-region transfer test (train in one area, test in another) was run: the
  geochemistry scores are standardised within each area, so the features are not comparable between projects.
- The peak memory figure is the process high-water mark; the Boundary run was an earlier process whose peak was
  not captured (a bug in the harness, since fixed).
