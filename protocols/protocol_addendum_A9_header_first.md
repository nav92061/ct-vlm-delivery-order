# Protocol addendum A9: header first, image second (written before any A9 call)

Question: is the second-turn effect "the header wins after commitment" or "whatever arrives last wins"?
Design: turn 1 = the header-only prompt carrying the same FALSE header used in the conflict conditions (no image);
the model answers. Turn 2 = the image of the same pair (aligned or native render, same description sentences as the
image-only prompt), "Here is the image of the same pair", and the standard "Give your final decision" instruction.
60 pairs x 4 variants: 240 turn-1 calls (text only) and 480 turn-2 calls (2 views) per model.
Models: Nemotron-3.5-VL-120B, Claude Haiku 5.5, Claude Sonnet 5.5 (settings as in their main runs).
Endpoint: among cases whose turn-1 answer followed the false header (wrong decision) AND that the model decided
correctly from the image alone (image-only condition), the share overturned by the image in turn 2. Compare with the
image-first reversal rate (share of correct image-only decisions overturned by the header in turn 2).
Interpretation fixed in advance: similar high rates in both directions -> recency ("last evidence wins"); image
rarely overturning the header -> asymmetric deference to the header.
Budget: Haiku cap $8.0 accounted (10x list); Sonnet cap $18.5 accounted (1.5x list). Worst-case project Claude list
total < $13.2.
