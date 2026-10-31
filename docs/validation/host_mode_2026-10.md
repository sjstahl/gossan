# Empirical host layer for the knowledge score (October 2026)

`scripts/validate_host_mode.py` ran the prospectivity model twice on each of the four BC areas of
`bc_regions_2026-10.md`, on copies of the same harvested layers, with the same settings and seed:

* **named**: `knowledge.host_mode: named` with no host units named, so the knowledge score has no host layer (the
  score as validated in the October note);
* **empirical**: `knowledge.host_mode: empirical`, the host layer learned from the occurrences and relearned inside every
  spatial fold from the training blocks only (2 km gap around each test block, 5 folds).

The forest is identical in both runs (checked), so only the knowledge score changes. Settings are the current defaults
(`occurrence_class: auto`, so metallic occurrences where deposit-type codes exist), which is why the forest and "named"
figures differ a little from the October note, which used all occurrences. The raw numbers are in `host_mode_*.json`.

## Results (held-out blocks, 5 folds, chance is 0.5 for AUC and 5% for capture)

| Region | Occurrences | Knowledge AUC named → empirical | Paired difference (± SD over folds, folds better) | Effort-matched AUC named → empirical | Paired difference | Top-5% capture | Host layer AUC held out / in-sample |
|---|---|---|---|---|---|---|---|
| Boundary | 375 | 0.800 → 0.837 | +0.037 ± 0.051 (4/5) | 0.658 → 0.680 | +0.022 ± 0.051 (3/5) | 27% → 31% | 0.71 / 0.83 |
| East Kootenay, wide | 380 | 0.673 → 0.772 | +0.099 ± 0.064 (4/5) | 0.571 → 0.602 | +0.032 ± 0.053 (4/5) | 13% → 22% | 0.70 / 0.74 |
| Quesnel | 356 | 0.707 → 0.758 | +0.051 ± 0.065 (4/5) | 0.649 → 0.695 | +0.047 ± 0.054 (4/5) | 15% → 32% | 0.66 / 0.72 |
| Stewart | 710 | 0.749 → 0.771 | +0.022 ± 0.034 (4/5) | 0.621 → 0.627 | +0.007 ± 0.032 (3/5) | 19% → 23% | 0.66 / 0.64 |

Forest, same runs (effort-matched AUC / plain AUC): Boundary 0.71 / 0.87, East Kootenay 0.59 / 0.75, Quesnel 0.66 / 0.73,
Stewart 0.62 / 0.74.

## What it shows

1. **The direction is the same everywhere.** Every one of the eight mean differences (plain and effort-matched AUC, four
   regions) is positive, and top-5% capture rises in all four. The held-out host layer alone ranks the held-out occurrences
   at 0.66 to 0.71, so it carries real information about ground the training blocks never saw.
2. **The size is modest and the evidence is thin.** Plain AUC gains +0.02 to +0.10. Effort-matched, which is the fairer
   test, the gains are +0.01 to +0.05, and at Stewart essentially nil. With five folds the mean difference is 1.4 to 3.5
   standard errors from zero for plain AUC and 0.5 to 1.9 for effort-matched; and the folds share training data and
   neighbouring ground, so they are not independent. Read these as consistent, not proven.
3. **The in-sample figure overstates it, as expected.** The host layer's own AUC over every cell (0.64 to 0.83) is above its
   held-out value except at Stewart, the largest gap being Boundary (0.83 against 0.71). That gap is what the old, circular
   check would have reported as skill.
4. **The knowledge score is no longer an independent second opinion.** It now uses the occurrences, through the same unit
   contrast the forest's top feature uses. Its rank correlation with the forest rose at Boundary (0.65 → 0.72), East Kootenay
   (0.43 → 0.59) and Quesnel (0.57 → 0.65), and fell slightly at Stewart (0.67 → 0.65). Top-5% overlap with the forest moved
   both ways (0.29 → 0.43, 0.16 → 0.14, 0.08 → 0.07, 0.16 → 0.04). If the point of the score is a check that does not learn
   from the occurrences, keep `named`; the empirical mode trades that independence for a modest gain in skill.
5. **With it, the knowledge score is ahead of the forest in three of four regions on plain AUC** (East Kootenay 0.77 against
   0.75, Quesnel 0.76 against 0.73, Stewart 0.77 against 0.74) and behind at Boundary (0.84 against 0.87). Effort-matched it is
   ahead at East Kootenay, Quesnel and Stewart by 0.01 to 0.04 and behind at Boundary (0.68 against 0.71). Differences of a few
   hundredths are within the fold-to-fold spread.

## Decision

`host_mode` stays `named` by default. The empirical mode is validated as non-circular and consistently, modestly helpful on
these four areas, none of which were used to choose its settings (`HOST_FAV_MAX` and the 0.2 floor are the stated starting
points). Making it the default would also change what the knowledge score means (point 4), so it is left to the user's choice.
Before recommending it for a new area it should be checked on ground outside these four.

## Limits

* Five folds on blocks of about 20 km; the 2 km gap is the project default, and large bedrock units still span the gap, so a
  unit that holds a camp in the training blocks can rank neighbouring held-out ground in the same unit. That is the intended
  generalisation, but it is spatially local, as the forest's skill is.
* One seed. Both runs use the same folds, so the paired differences are fair, but a different split could move them.
* The generic preset gives the host layer weight 3 (the largest, with geochemistry); a different weight was not tried.
