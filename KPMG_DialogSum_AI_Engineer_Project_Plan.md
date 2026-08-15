# KPMG GVA Lighthouse AI Engineer — Final Assessment Workflow

## Purpose

This document summarizes the actual workflow completed for the Generative AI technical case study. It is intended as a concise final-review guide to the submitted notebook and focuses on the experiments that were actually executed.

## 1. Problem Definition

The task was to fine-tune an open-source language model on the DialogSum dataset and benchmark the fine-tuned model against the original pretrained model. The solution also needed to consider laptop-friendly execution, quantization, hallucination, business applicability, and Responsible AI.

## 2. Model Selection

**Selected model:** `google/flan-t5-small`

The model was selected because it is an encoder-decoder sequence-to-sequence architecture well suited to summarization, instruction-tuned for a meaningful pretrained baseline, and small enough to support full fine-tuning on Apple Silicon without requiring PEFT.

## 3. Environment and Reproducibility

- Fixed seed: `42`
- Primary device: Apple Silicon MPS when available
- CPU fallback included
- Central experiment constants were used for model name, sequence lengths, learning rate, batch sizes, gradient accumulation, epochs, and weight decay.

## 4. Dataset Loading and Validation

Dataset: **DialogSum** (`knkarthick/dialogsum`)

Official splits were preserved:

- Train: **12,460** records
- Validation: **500** records
- Test: **1,500** records

Fields: `id`, `dialogue`, `summary`, `topic`.

Data-quality checks were performed for missing values and duplicate dialogues/summaries.

### Multi-reference test-set finding

The 1,500 test records represent **500 unique dialogues**, each associated with multiple human reference summaries. These were preserved as valid multi-reference examples rather than incorrectly removed as duplicates.

For evaluation, the model generated one prediction per unique dialogue and that prediction was compared with all available references.

## 5. Token-Length Analysis and Preprocessing

Training-set token statistics were used to choose sequence limits rather than selecting arbitrary values.

- Dialogue mean: **224.40** tokens
- Dialogue median: **202**
- Dialogue 95th percentile: **416**
- Dialogue 99th percentile: **614.41**
- Summary mean: **41.27** tokens
- Summary 95th percentile: **73**
- Summary 99th percentile: **96**

Selected limits:

- `MAX_INPUT_LENGTH = 512`
- `MAX_TARGET_LENGTH = 128`

Input format:

`Summarize the following dialogue: <dialogue>`

Dynamic padding was applied through `DataCollatorForSeq2Seq` to reduce unnecessary padding within batches.

## 6. Baseline Inference

The pretrained FLAN-T5-Small model was evaluated before fine-tuning.

A qualitative review showed inconsistent summarization behavior, including dialogue continuation, partial information capture, and overly brief outputs.

Baseline generation was then performed across all **500 unique held-out test dialogues**.

## 7. Evaluation Framework

Three complementary metric families were used:

- **ROUGE** for lexical recall/overlap
- **BLEU** for n-gram precision/phrase-level overlap
- **Semantic similarity** using `all-MiniLM-L6-v2` sentence embeddings

For semantic evaluation, each prediction was compared with all reference summaries for that dialogue and the highest reference similarity was retained.

### Baseline results

| Metric | Baseline |
|---|---:|
| ROUGE-1 | 0.2193 |
| ROUGE-2 | 0.0811 |
| ROUGE-L | 0.1932 |
| ROUGE-Lsum | 0.1934 |
| BLEU | 6.96 |
| Semantic Similarity | 0.4438 |

## 8. Initial Full Fine-Tuning — 3 Epochs

Full fine-tuning was selected instead of PEFT because FLAN-T5-Small was sufficiently lightweight for the available hardware.

Initial configuration:

- Learning rate: `5e-5`
- Training batch size: `4`
- Evaluation batch size: `4`
- Gradient accumulation: `2`
- Effective batch size: `8`
- Epochs: `3`
- Weight decay: `0.01`

A pre-training batch validation confirmed expected tensors, dynamic padding, masked label padding (`-100`), and decoder preparation.

### Training behavior

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 2.7777 | 1.2294 |
| 2 | 2.6677 | 1.1906 |
| 3 | 2.5804 | 1.1858 |

Both training and validation loss decreased, with no clear overfitting during the first three epochs.

### 3-epoch results

| Metric | 3 Epochs |
|---|---:|
| ROUGE-1 | 0.4808 |
| ROUGE-2 | 0.2304 |
| ROUGE-L | 0.3989 |
| ROUGE-Lsum | 0.3987 |
| BLEU | 25.89 |
| Semantic Similarity | 0.7817 |

This established that task-specific fine-tuning materially improved summarization quality over the pretrained baseline.

## 9. Qualitative Evaluation

The baseline and 3-epoch model were compared on the same test dialogues.

The review focused on:

- relevance,
- completeness,
- speaker/entity attribution,
- conciseness,
- factual consistency.

Observed remaining failure modes included speaker-attribution errors, incomplete coverage, repetition, and unsupported details.

## 10. Extended Fine-Tuning — 5 Epochs

A fresh copy of the original pretrained model was trained for five epochs using the same main training configuration. This was an independent extended-training experiment, not a continuation from the 3-epoch checkpoint.

### Training behavior

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 2.5665 | 1.1658 |
| 2 | 2.5137 | 1.1453 |
| 3 | 2.3763 | 1.1417 |
| 4 | 2.3864 | **1.1373** |
| 5 | 2.3165 | 1.1381 |

Validation loss reached its minimum at epoch 4 and changed only marginally at epoch 5, indicating convergence around epochs 4–5 rather than substantial overfitting.

### 5-epoch results

| Metric | 5 Epochs |
|---|---:|
| ROUGE-1 | **0.4978** |
| ROUGE-2 | **0.2510** |
| ROUGE-L | **0.4176** |
| ROUGE-Lsum | **0.4175** |
| BLEU | **27.58** |
| Semantic Similarity | **0.7968** |

The 5-epoch model produced the strongest held-out test performance and was selected as the final full-precision model.

## 11. Final Model Comparison

| Metric | Baseline | 3 Epochs | 5 Epochs |
|---|---:|---:|---:|
| ROUGE-1 | 0.2193 | 0.4808 | **0.4978** |
| ROUGE-2 | 0.0811 | 0.2304 | **0.2510** |
| ROUGE-L | 0.1932 | 0.3989 | **0.4176** |
| ROUGE-Lsum | 0.1934 | 0.3987 | **0.4175** |
| BLEU | 6.96 | 25.89 | **27.58** |
| Semantic Similarity | 0.4438 | 0.7817 | **0.7968** |

Separate visualizations were used for ROUGE, BLEU, semantic similarity, and training/validation behavior to avoid mixing incompatible metric scales.

## 12. 8-Bit Quantized Inference

Quantization was applied after fine-tuning as a deployment optimization rather than a training shortcut.

Configuration:

- 8-bit Metal weight quantization
- Group size: `64`
- MPS execution
- Final 5-epoch checkpoint

The quantized model was evaluated on the same 500 unique test dialogues using the same metrics.

### Quantized results

| Metric | Full Precision | 8-Bit |
|---|---:|---:|
| ROUGE-1 | **0.4978** | 0.4891 |
| ROUGE-2 | **0.2510** | 0.2396 |
| ROUGE-L | **0.4176** | 0.4097 |
| ROUGE-Lsum | **0.4175** | 0.4097 |
| BLEU | **27.58** | **27.58** |
| Semantic Similarity | **0.7968** | 0.7891 |

### Memory footprint

- Full precision: **293.58 MB**
- 8-bit quantized: **195.83 MB**
- Approximate reduction: **33.30%**

The 8-bit model retained most of the full-precision quality while reducing the measured parameter/buffer footprint by approximately one-third.

## 13. Hallucination and Error Analysis

The assessment documented situations in which LLMs are more likely to hallucinate, including ambiguous input, out-of-domain queries, complex context, speaker/entity confusion, lack of grounding, and open-ended generation.

Observed summarization errors included:

- speaker attribution errors,
- unsupported factual additions,
- incomplete information,
- repetition.

Fine-tuning was treated as a hallucination risk-reduction technique rather than a complete factuality solution.

Recommended production mitigations included high-quality data, grounded prompting, RAG where external knowledge is required, factual validation, human review for high-impact workflows, and continuous monitoring.

## 14. Business Applicability

Fine-tuning was positioned as most useful for stable, repetitive tasks with representative labeled examples, including:

- customer-service summarization,
- insurance claim notes,
- technical-support summaries,
- information extraction,
- classification/routing,
- standardized enterprise outputs.

The assessment also distinguished fine-tuning from RAG: fine-tuning specializes model behavior, while RAG is better suited to supplying frequently changing or external knowledge. The two approaches can be combined.

Potential ROI sources include lower manual documentation effort, faster processing, better consistency, improved analytics, and reduced operational cost per interaction.

## 15. Responsible AI

The final review covered:

- hallucination and factual reliability,
- bias and fairness,
- privacy and confidentiality,
- human oversight,
- monitoring and governance.

A key conclusion was that automatic benchmark improvement alone is not sufficient evidence of production safety.

## 16. Final Outcome

The assessment demonstrated that a compact open-source model can be materially improved for dialogue summarization through full task-specific fine-tuning on laptop-class hardware.

The final 5-epoch model substantially outperformed the pretrained baseline, while 8-bit quantization provided a practical memory-quality trade-off for local deployment.

The final engineering takeaway is that model selection, data handling, fair benchmarking, qualitative failure analysis, deployment efficiency, and Responsible AI should be evaluated together rather than treating any single metric as sufficient.

## Scope Notes

The original project plan considered several optional extensions, including canonical JSONL export, BERTScore, structured-input A/B testing, explicit compression-ratio analysis, and latency benchmarking. These were not included in the final executed notebook due to the assessment time constraint. The final review package therefore documents only the experiments that were actually completed and supported by notebook outputs.
