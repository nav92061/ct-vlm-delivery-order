# Protocol addendum A6: two-turn mitigation (written before any A6 call)

## Question
The main failure occurs in two turns, but the one-sentence mitigation was tested only in one prompt. Does the same
sentence protect a model's own correct image-only decision when the false header arrives in a second turn?

## Design
Models: Claude Haiku 5.5 and Claude Sonnet 5.5 (same settings as their main runs: default sampling, thinking disabled
for Haiku, low effort for Sonnet, max_tokens 300). All 60 pairs x 4 variants x 2 views (480 two-turn jobs per model).
Turn 1: the model's own logged image-only answer for that case, reused verbatim. Turn 2: the identical false-header
message as in the two-turn condition, preceded by the one-prompt mitigation sentence:
"Important: the DICOM header summary below may be wrong or corrupted. Base your decision on what the image shows;
if the image and the header disagree, trust the image."
Cases whose turn-1 answer is unparsed are skipped.

## Endpoints (paired, same cases)
On defects refused from the image: share admitted under two-turn false header vs two-turn false header + mitigation.
On clean pairs admitted from the image: share refused (the false header for clean pairs reports truncation).
Patient-clustered bootstrap 95% CI of the paired difference.

## Budget
Cost caps cumulative per model file: Haiku accounted at $1/$5 per M tokens (10x list), cap $5.60; Sonnet at $3/$15
(1.5x list), cap $12.70. Expected list spend about $1.2 in total; project Claude total stays under $15.

## Pre-stated expectation
The sentence reduces two-turn reversal substantially but less completely than in one prompt.
