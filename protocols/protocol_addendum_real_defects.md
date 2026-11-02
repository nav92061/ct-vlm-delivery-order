# Protocol addendum A2: naturally occurring coverage mismatches (frozen 2026-10-28, before any download or model call)

## Question
Do the main-grid findings (image-only silent failure; header override; prompt mitigation) hold on real NLST
reconstruction pairs whose z-coverage genuinely differs, with no synthetic edits?

## Cases
All 24 pairs in real_pairs_geometry.csv (same NLST study, two reconstructions) with header-computed z-IoU < 0.95.
None were used in the main grid (whose pairs were selected with z-IoU = 1). Ground truth for every pair is
REFUSE under the frozen admission rule ("same z-range and no missing slices"). Defect size is recorded as
1 - z-IoU. Mid-coronal planes are extracted with the unchanged extract_pairs.py logic; renders use the unchanged
study_render.py (aligned and native views). There are no synthetic edits.

## Conditions (unchanged prompt templates)
image_only (aligned, native); image+correct_header (aligned); image+false header (aligned, native), where the false
header states that Series B has Series A's z-range and slice count (the "reassuring" direction); and the same two
false-header conditions with the addendum A1 mitigation sentence. That gives 7 prompts per pair.

## Models and settings
Nemotron-3.5-VL-120B, Llama-3.2-11B-V, Llama-3.2-90B-V, Gemma-4-31B, Muse-Glimmer-30B (NVIDIA API, temperature 0,
the main-grid token limits) and MedGemma-1.5-4B (Modal, greedy decoding). Results go to real_results.jsonl.

## Endpoints
Silent failure (share admitted) per model and condition, with a patient-level bootstrap 95% CI. The paired
difference compares the false header with image only (same view), and mitigation with the false header.
Because every real case is defective, false refusal is taken from the 60 main-grid clean pairs. Results are
reported stratified by defect size (1 - z-IoU < 0.15 vs >= 0.15).

## Pre-stated expectation
The ranking of models and the direction of the header effect match the synthetic grid. Because some real
mismatches are small (5-10%), image-only silent failure may be higher than on the synthetic defects.
