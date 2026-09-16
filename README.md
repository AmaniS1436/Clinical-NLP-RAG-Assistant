# Clinical NLP & RAG Assistant

A medical-text classification and Retrieval-Augmented Generation (RAG) pipeline, built end to end in a single Google Colab notebook (NVIDIA T4 GPU, free tier).

> **Scope disclaimer.** This is an AI-engineering portfolio project, not a clinical or diagnostic system. It has not been clinically validated and must never be used as a source of medical advice. All results below are exactly what was measured — nothing extrapolated, nothing invented — and known limitations are documented rather than hidden.

## What's inside

The notebook covers two connected parts:

**Part 1 — Medical text classification**
A TF-IDF + Logistic Regression baseline compared against a fine-tuned domain-specific Transformer (BioBERT), on the [`TimSchopf/medical_abstracts`](https://huggingface.co/datasets/TimSchopf/medical_abstracts) dataset (5 medical condition categories).

**Part 2 — Retrieval-Augmented Generation**
A small medical knowledge base built from the WHO Cancer Fact Sheet, indexed with sentence embeddings and ChromaDB, queried through FLAN-T5-base, with an explicit distance-based **abstention mechanism** for questions the knowledge base doesn't cover.

## Results

### Classification

| Model | Test Accuracy | Test Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 81.8% | 80.9% |
| BioBERT (fine-tuned) | 86.9% | 85.8% |

Both models struggle most with the **general pathological conditions** class — it's the broadest, most heterogeneous category and shares vocabulary with the other four. BioBERT's biomedical pretraining narrows this gap but doesn't fully resolve it.

### RAG abstention

A manual benchmark of 10 questions (6 answerable from the knowledge base, 4 deliberately out of scope) was used to validate the abstention mechanism:

- Accuracy / Precision / Recall / F1: **1.00** on this 10-question benchmark

This measures that the abstention rule worked correctly on this small, manually built set — it is **not** a claim of general RAG robustness. A larger, more diverse benchmark is the top item under future work.

### A documented limitation

Even when retrieval returns fully relevant context, the lightweight generator (FLAN-T5-base) can produce an **incomplete** answer — e.g. for *"How can cancer mortality be reduced?"*, it generated only *"When cases are detected and treated early"* despite the retrieved context also covering treatment and care. Retrieval and generation were evaluated separately specifically to catch failures like this one: good retrieval does not guarantee good generation.

## Pipeline overview

```text
Medical Abstracts Dataset → Data Cleaning → Train/Val/Test
        → TF-IDF Baseline & BioBERT → Evaluation & Comparison

WHO Cancer Fact Sheet → Extraction → Cleaning → Semantic Chunking
        → Embeddings (MiniLM) → ChromaDB
        → User Question → Retrieval → Distance Threshold
              → Reject → Abstain
              → Accept → Context → FLAN-T5-base → Answer + Sources
```

## Data quality note

Before any training, the raw dataset was checked for train/test leakage and label conflicts: 988 exact duplicate texts were found across the original train/test splits, and 2,929 unique texts carried conflicting labels. Both were removed, taking the dataset from 14,438 rows down to a clean, unambiguous **8,298 documents** used for all subsequent training and evaluation.

## Stack

Python · Pandas · NumPy · scikit-learn · Matplotlib · Hugging Face `transformers` / `datasets` · PyTorch · Sentence-Transformers · ChromaDB · BeautifulSoup · Google Colab (T4 GPU)

## Repository contents

- `Clinical_NLP_RAG_Assistant.ipynb` — the full, documented notebook (setup → classification → RAG → evaluation → conclusion)

## Possible next steps

- Larger, more diverse RAG evaluation benchmark
- Semantic (not just abstain/keyword) evaluation of generated answers
- Reranking on top of vector retrieval
- Comparison of multiple embedding models
- Stronger instruction-tuned generator
- Faithfulness / answer-completeness metrics
- FastAPI service + Docker packaging for a deployable demo



