# Four-page technical report audit and revision

**Date:** 13 November 2026  
**Evidence snapshot:** `Agnuxo1/solar-filament-segmentation-2026`, commit `83835e101bc7fa75479e680fc3585be27b530fc5`  
**Scope:** document correction and verification; no new training, hidden-test evaluation, Kaggle submission, organizer-form submission, or remote repository edit.

## Decision

The existing four-page report required a replacement. Its numerical results were broadly consistent with the recorded experiments, but its template, semantic-model description and public release link contained material problems. The revised PDF preserves the historical official result and measured figures, corrects those problems, and incorporates the later recorded audits through P104.

## Original artifact

- Repository path: `docs/submission/TECHNICAL_REPORT_4P_20261112.pdf`.
- Original SHA-256: `68dafbb8feb7ebeadf8e3dc8273ce8c0bd749bcd3024b477ddef8e0694e9cdc0`.
- Four A4 pages, single-column layout; readable, with no visual clipping.
- No corresponding complete LaTeX source was located in either the participant's public or development repository. The older development report draft was not the source of this PDF.

## Corrections

| Issue | Evidence and revision |
|---|---|
| Original single-column report did not follow the organizers' template | The official public Overleaf compile log identifies `acmart`, `sigconf`, and 9-point text. Rebuilt with that format, template metadata, fixed introduction opening, subtitle and acknowledgment. |
| Semantic ensemble incorrectly implied an EfficientNetV2-S member | The accepted H3 manifest and implementation contain four ResNet34 U-Nets. EfficientNetV2-S belongs to one of the two crop refiners. The revised method states this explicitly. |
| Original link pointed to the private development repository | Replaced by the public `Agnuxo1/solar-filament-segmentation-2026` repository. |
| Reproduction claim lacked later provenance results | Added P93's corrected reference identity, P94's exact 1,241-row replay from frozen crops, P95's bounded crop-stage result, P99's consensus result, P102's 14/15-row pilot, and P104's static checks. P103 is described as preregistered, without a fabricated completion. |
| Small-filament conclusion could be read from favorable calibration alone | Added the negative held-out effects, three additional F1 seeds in P83, and P84's ten correlated F1–F4 reports (three positive, seven negative). No promotion or new official score is claimed. |
| Supplementary splitting/merging diagnostics could be confused with the organizer's definitions | Defined the participant's coverage thresholds explicitly and identified the figures as H6 held-out diagnostics. |
| Historical no-component fallback could be conflated with accepted H9b | Described the notebook audited at P104 in the past tense and stated that independently regenerated output does not inherit PQ 0.39 without its own verification. |

## Preserved facts

- Official historical public PQ: **0.39**, accepted submission **58112557**, 8 November 2026.
- Accepted CSV SHA-256: `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`.
- Four semantic ResNet34 U-Nets, four original Mask R-CNN models, two refiners (ResNet34 and EfficientNetV2-S).
- Original H6 example panel and metric panels, extracted losslessly from the existing PDF.
- Local cross-block measurements, confidence intervals, the unpassed +0.005 promotion threshold, and distinction between local measurements and official scores.

## Template provenance and build

Official template: https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe

The official publicly rendered log was observed using acmart 2.14, sigconf and 9-point type. The read-only editor did not expose all source and the download endpoint was restricted. The common preamble and fixed sections were recovered from attributed public copies and cross-checked; no competitor's method, measurements or figures were used. The source README records the exact public preamble commit and the second template copy.

The final PDF uses the unmodified current CTAN acmart class 2.21, with its standard sigconf geometry and font settings. The source ZIP includes that class and its upstream source/license attribution. Libertine, Inconsolata and NewTX fonts are embedded. The class version differs from the official project's observed version; the required class family, format and font size are retained. The PDF was not produced by editing the original raster pages.

## Verification

- **Exactly four pages**, 612 × 792 points (the standard acmart sigconf page size).
- All four final pages rendered and visually inspected.
- No overlapping headers, clipping, missing glyphs, unresolved references, or overfull TeX boxes.
- Tables and both measured figures are legible and attributed to the correct H6/H9b populations.
- Complete editable source and template provenance included.
- Verification manifest records the final file sizes and SHA-256 hashes.

Final PDF: `TECHNICAL_REPORT_4P_20261113.pdf`  
SHA-256: `6a289170b22e1fa06f328c0ebfad3e55686963c6ef81c5af3881986cb954d8bb`

## Remaining task boundaries

This document revision does not establish completion of the 180-image raw-JPEG P103 replay or a clean-environment execution of a newly revised portable notebook. It does not constitute an organizer-form receipt. Subsequent notebook fixes, measured runs and completed submissions require their own recorded evidence. Those limits are stated in the PDF itself.

The local files are ready for the participant's publication and submission workflow. No remote repository was changed by this audit.
