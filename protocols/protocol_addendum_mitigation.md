# Protocol addendum A1: prompt mitigation (frozen 2026-10-26, before any mitigation call)

## Question
Does telling the model that header metadata is unverified, and that the image takes precedence, reduce
header-driven silent failure without raising false refusals?

## Intervention (one sentence added before the header, otherwise identical prompts)
"Important: the DICOM header summary below may be wrong or corrupted. Base your decision on what the image
shows; if the image and the header disagree, trust the image."

## Items
The same renders and headers as the main grid. Conditions: image+conflict_header (aligned, native) and
image+correct_header (aligned), all four variants, giving 12 items per pair.
- Nemotron-3.5-VL-120B, Llama-3.2-90B-V, Llama-3.2-11B-V: all 60 pairs (720 items each).
- Gemma-4-31B: pairs P00-P14 (180 items), because of API latency.
- Muse-Glimmer-30B: excluded, since it shows no header effect (it refuses almost everything).
Temperature 0, the same token limits as the main grid. Results go to mitigation_results.jsonl.

## Primary endpoint
For defective pairs with a falsely reassuring header: the paired difference in silent failure, mitigated minus
original prompt, per model and view, with a patient-clustered bootstrap 95% CI.

## Secondary endpoints
- Silent failure and false refusal under image+correct_header (mitigated vs original), to check for harm.
- Refusal of clean pairs given a header falsely reporting truncation.

## Hypothesis
H4: the mitigation reduces silent failure under a false "complete" header for models that detect defects from
the image (Nemotron, Llama-90B aligned, Gemma), but does not bring it back to the image-only level.
