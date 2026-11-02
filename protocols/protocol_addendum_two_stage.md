# Protocol addendum A4: two-stage "image first, then header" mitigation (written before any two-stage call)

## Question
Does asking the model to judge the image before it sees the header reduce header-driven silent failure, without
any instruction about which source to trust? This separates information order from the explicit A1 instruction.

## Design (two-turn conversation, same model, temperature 0)
- Turn 1 (user): the unchanged image-only prompt and render (same view). Turn 1 (assistant): the model's own
  logged image-only answer from the main grid, reused verbatim. It is not regenerated, so stage 1 is exactly
  the main-grid image-only decision.
- Turn 2 (user): "Here is the DICOM header summary for the same pair: <header>. Give your final decision."
  followed by the unchanged FORMAT line. The header is the same falsely reassuring header (defective cases) or
  falsely alarming header (clean cases) used in the image+conflict_header condition. There is no sentence about
  which source to trust.
- Condition label: image+conflict_header+two_stage. Views: aligned and native.

## Models and items
- Nemotron-3.5-VL-120B: all 60 pairs, both views (480 calls).
- Gemma-4-31B: pairs P00-P14, both views (120 calls), because of API latency.
- Excluded: Llama-3.2-11B and MedGemma (do not detect defects from the image), Muse-Glimmer (refuses nearly
  all), Llama-3.2-90B (API unreachable).
- Cases whose logged stage-1 answer was UNPARSED are skipped and counted.

## Endpoints
Primary: paired change in silent failure on defective pairs, two-stage vs one-stage false header (same case,
same view), with a patient-clustered bootstrap 95% CI. Secondary: false refusal on clean pairs under the falsely
alarming header; the share of stage-1 REFUSE decisions reversed to ADMIT at stage 2; and a comparison with the
A1 instruction on the same cases.

## Pre-stated expectation
Two-stage ordering reduces silent failure relative to the one-stage false header, but by less than the explicit
A1 instruction, because nothing tells the model to prefer its own visual judgement.
