# Same evidence, different verdicts

Code, prompts, images and every model call log for:

> **N. Avelar.** *Same evidence, different verdicts: delivery of CT images and DICOM metadata changes VLM quality-control decisions.* Submitted to IEEE ISBI 2027.

## The finding in one paragraph

We asked vision–language models (VLMs) to act as an input check: given two CT reconstructions from the same scan, admit the pair only if both cover the same body region with no missing slices. Some pairs were defective (truncated, gapped or shifted), and some prompts included a **false DICOM header** saying the pair was complete. When the image and the false header came **together**, models were cautious and refused if either looked wrong. When the header came **after** the model had already answered from the image, models abandoned most of their correct detections. Reversing the order, the image overturned the header just as reliably: whichever source arrived second won.

| Defects detected from the image that were then admitted (aligned view) | Header with image | Header after the model's answer |
|---|---|---|
| Claude Haiku 5.5 | 2% | 91% |
| Nemotron-3.5-VL | 43% | 100% |
| Claude Sonnet 5.5 | 0% | 21% |

## Reproduce the headline numbers (one minute, no API keys)

```bash
pip install -r requirements.txt
python code/3_analysis/reproduce_headline.py
```

This reads the raw logs in `logs/` and prints the table above for both views. All other numbers in the paper come from the scripts in `code/3_analysis/` and are saved in `results/`.

## Repository layout

| Folder | Contents |
|---|---|
| `paper/` | The submitted paper (PDF) |
| `data/` | Case tables (`main_pairs.csv`, `real_pairs.csv`, `study_cases.csv`, `real_cases.csv`) and the 528 rendered image pairs (`data/renders/`) |
| `prompts/` | Every prompt sent to the API models (`jobs.json`), including two-turn follow-ups, and the real-pair items |
| `code/1_data_and_renders/` | Pair selection, defect simulation, rendering, and the pixel-only reference check |
| `code/2_model_calls/` | Scripts that sent prompts to each model and logged the answers |
| `code/3_analysis/` | Scripts that compute every number in the paper from the logs |
| `code/4_figures/` | Figure scripts and the editable Fig. 1 (`.drawio`) |
| `logs/` | Raw model answers, one JSON line per call, with timestamp, model ID, settings, full response text and parsed decision |
| `protocols/` | The plan for each experiment, written before that experiment was run |
| `results/` | Analysis tables behind the paper's figures and text |

## Data

Images are derived from the **National Lung Screening Trial (NLST)** CT collection in the **NCI Imaging Data Commons**, released under **CC BY 4.0**. All data are de-identified and public. Raw DICOM files are not redistributed here; `data/main_pairs.csv` and `data/real_pairs.csv` identify every series used (columns `sa`, `sb` are SeriesInstanceUIDs), so they can be re-downloaded with `idc-index`.

## Models

| Model | Access | Settings |
|---|---|---|
| Nemotron-3.5-VL-120B, Gemma-4-31B, Llama-3.2-11B/90B-Vision, Muse-Glimmer-30B | NVIDIA API catalog | temperature 0 |
| MedGemma-1.5-4B, Qwen3-VL-8B | open weights, one A10G GPU | greedy decoding |
| Claude Haiku 5.5, Claude Sonnet 5.5 | Anthropic API | default sampling (temperature not accepted); thinking disabled (Haiku), low effort (Sonnet) |

Exact model identifiers and per-call settings are recorded in every log line. Hosted models may change over time.

## Re-running model calls (optional, costs money)

The scripts in `code/2_model_calls/` read API keys only from environment variables (`NVIDIA_API_KEY` or `K`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`); no keys are stored in this repository. `run_frontier.py` supports OpenAI, Anthropic, Google and NVIDIA endpoints, a dry-run `--provider mock` mode, and a hard `--max-cost` cap. Run with `--limit 20` first.

## AI use

A generative AI assistant (Claude, Anthropic) wrote the evaluation and analysis code, ran the model API calls under the author's direction, and drafted the paper text. The author directed the study, chose which experiments to run and how to frame the results, and reviewed all code, results and text. All evaluated models, including Claude, received the same frozen prompts and analysis.

## Citation

See `CITATION.cff`. Until the paper is published, please cite it as a submission to IEEE ISBI 2027.

## Licence

Code: MIT (see `LICENSE`). Rendered images and derived data: CC BY 4.0, following the NLST/IDC source licence.
