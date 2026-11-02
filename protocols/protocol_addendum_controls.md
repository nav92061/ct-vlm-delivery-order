# Protocol addendum A5: reviewer-requested controls (written before any A5 call)

Model: Nemotron-3.5-VL-120B (temperature 0, main-grid token limit). All 60 pairs x 4 variants. Same renders and
task prompt as the main grid unless stated. Output: controls_results.jsonl.

## C1 raw-field header (one stage, aligned + native; 480 calls)
The header is given as raw DICOM-style fields per series, with no derived "interior gaps" or "same range"
statement: first/last slice z from ImagePositionPatient, NumberOfSlices, SliceThickness and SpacingBetweenSlices.
The false version uses the same numbers as the main-grid conflicting header (B reported with A's z-range and a
contiguous slice count; for clean pairs, the main-grid falsely truncated values). Question: does deference
survive when the model must compute the verdict itself?

## C2 neutral-text control (one stage, aligned + native; 480 calls)
Non-geometric metadata of similar length in place of the header (Modality, BodyPartExamined, StudyDescription,
Manufacturer, PatientPosition), true for every case. Question: does the presence of any text change decisions?

## C3 two-turn correct header (480 calls) and C4 two-turn neutral (480 calls)
Same as addendum A4 (turn 1 = the logged image-only answer reused verbatim), but turn 2 contains the TRUE header
(C3) or no new information: "Please confirm your final decision." (C4). Question: are A4 reversals caused by the
false header or by a follow-up turn as such (sycophancy)?

## C5 native view without z labels (image only; 240 calls)
The native renders regenerated with tick labels removed (same pixels otherwise). Question: how much of
native-view detection depends on reading z labels?

## Endpoints
SFR (defective) and FRR (admissible), each paired against the matching main-grid or A4 condition, with a
patient-clustered bootstrap 95% CI. Unparsed answers are reported, with bounds counting them as admit or refuse.

## Pre-stated expectations
C1: deference persists but is lower than with the summary header. C2: no material change from image only.
C3: decisions follow the true header (few reversals of correct refusals). C4: few reversals. C5: native-view
detection drops for z-shifts (equal panel heights) more than for truncations.


## Deviation (recorded before analysis of C1/C2/C5)
API rate limiting (HTTP 429) allowed ~4 successful calls/min. C3 and C4 were stopped after partial coverage (all completed calls are reported); remaining capacity went to C1 (all 60 pairs) and C5 (all 60 pairs). C2 was limited to pairs P00-P14.

## Deviation 2 (2026-10-29, before final analysis)
C3 and C4 had been stopped after partial coverage (interim results, 0 reversals in both, had been seen). To remove any
dependence of the stopping point on the results, both are being run to completion on all eligible cases
(stage-1 answer parsed), after C1, C5 and the C2 subset finish. The final analysis uses all completed cases.
