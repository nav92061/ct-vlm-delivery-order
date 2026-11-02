# Protocol addendum A3: pixel-only reference baseline (written 2026-10-28, before the baseline is run on any case)

## Question
Is the coverage defect recoverable from the rendered image pixels alone, i.e. is the task solvable from the
exact inputs the VLMs receive, without the header?

## Method (no learning, no header, no OCR of the z-tick labels)
Input: the same JPEG renders shown to the VLMs (aligned and native views; main grid and 24 real pairs).
1. Locate the Series A and Series B panels as the column runs that differ from the uniform grey canvas (value 90).
2. Aligned view: for each panel, mark each image row as "data" if it contains any pixel > 40 (no-data rows are
   uniform black). Score = fraction of rows whose data/no-data status differs between A and B, divided by the
   number of A data rows.
3. Native view: take the row-mean intensity profile of each panel (both panels are drawn at the same pixels per
   mm). Score = max(|h_A - h_B| / h_A, |d*| / h_A), where h is the panel's data height and d* is the vertical
   offset of B relative to A that maximises the normalised cross-correlation of the two profiles (searched
   over +/-40% of h_A, overlap at least 50%).
4. Decision: REFUSE if score > 0.03, else ADMIT. The threshold is fixed a priori: it is below the smallest
   synthetic defect (5%) and the smallest real mismatch (5%), and above one-slice rounding.

## Endpoints
SFR and FRR per view (main grid: 240 cases; real pairs: 24, all defective), with the same patient-clustered
bootstrap as the VLMs; plus threshold-free AUC of the score. By construction the baseline does not read the
header, so its decisions are identical under the image-only, correct-header and false-header conditions.

## Interpretation stated in advance
High accuracy means the defect is visible in the provided pixels, so VLM silent failure reflects the model, not
an invisible defect. The baseline is a reference, not a proposed method: it is specific to this rendering.
