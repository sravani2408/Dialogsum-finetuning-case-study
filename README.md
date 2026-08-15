# KPMG AI Engineer — Final Review Package

This folder contains the final materials for reviewing the completed Generative AI case study.

## Folder Contents

### Submission/
- `KPMG_Generative_AI_Case_Study_Final.ipynb` — final notebook. Code and outputs are preserved from the submitted notebook; only minor Markdown ordering/heading cleanup was applied for review clarity.

### Documentation/
- `KPMG_Assessment_Final_Workflow.md` — concise record of the workflow actually completed, including model choice, dataset handling, baseline, fine-tuning, evaluation, quantization, hallucination analysis, business interpretation, and Responsible AI.

### Reference/
- `GenerativeAI_Case_Study_2026.pdf` — original case-study brief.

## Final Results Snapshot

| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Semantic Similarity |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2193 | 0.0811 | 0.1932 | 6.96 | 0.4438 |
| 3 Epochs | 0.4808 | 0.2304 | 0.3989 | 25.89 | 0.7817 |
| 5 Epochs | **0.4978** | **0.2510** | **0.4176** | **27.58** | **0.7968** |
| 8-Bit Quantized | 0.4891 | 0.2396 | 0.4097 | 27.58 | 0.7891 |

8-bit quantization reduced the measured model parameter/buffer footprint from **293.58 MB** to **195.83 MB**, an approximate **33.30% reduction**.

## Review Note

The workflow document intentionally reflects the experiments that were actually executed. Optional ideas from the earlier project plan that were not completed are identified as out of scope rather than presented as completed work.
