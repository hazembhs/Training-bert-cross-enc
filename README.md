# Training BERT Cross Encoder for Ranking

This repository provides an implementation of a BERT-based cross-encoder model for information retrieval tasks, specifically for ranking query-document pairs. The project includes a `VanillaBertRanker`, which leverages the pretrained BERT model for learning query-document relevance scores.

---

## Table of Contents

1. [Features](#features)
2. [Setup](#setup)
3. [Usage](#usage)
    - [Training](#training)
    - [Validation](#validation)
4. [Components](#components)
5. [Performance Metrics](#performance-metrics)
6. [Acknowledgments](#acknowledgments)

---

## Features

- **BERT-based Architecture**: Fine-tuning the `bert-base-uncased` model for ranking tasks.
- **Custom Tokenizer**: Handles tokenization and special token management.
- **Subbatching**: Efficient handling of long documents through subbatching for memory optimization.
- **Flexible Parameter Tuning**: Supports gradient accumulation, learning rate adjustment, and patience for early stopping.
- **Scoring and Ranking**: Evaluates Mean Reciprocal Rank (MRR@100) for validation.

---

## Setup

1. **Dependencies**:
   - Python 3.8+
   - PyTorch 1.10+
   - tqdm
   - `pytorch-pretrained-bert`

   Install requirements:
   ```bash
   pip install -r requirements.txt


1. **usage**: 
python main.py \
    --model vanilla_bert \
    --datafiles data/train.jsonl \
    --qrels data/qrels.train \
    --qrels_valid data/qrels.valid \
    --train_pairs data/train_pairs.jsonl \
    --valid_run data/valid_run.jsonl \
    --initial_bert_weights weights/bert_base_uncased.p \
    --model_out_dir model_out/

## Components

### Model

- **BertRanker**:  
  Base class for rankers, implementing core BERT encoding and tokenization logic.

- **VanillaBertRanker**:  
  Extends `BertRanker` with a dropout layer and a classification head for relevance scoring.

### Tokenization

- The `tokenize` method splits text into BERT-compatible tokens and converts them into IDs using the vocabulary.

### Subbatching

- Handles documents longer than BERT's positional encoding limit by splitting them into subbatches.

### Training

- Implements gradient accumulation and adaptive learning rates for BERT and non-BERT parameters.

### Validation

- Computes **MRR@100** as the primary metric.

---

## Performance Metrics

The model evaluates performance using the **MRR@100** metric during validation. It measures the ranking quality, focusing on the first relevant result.

---

## Acknowledgments

This implementation is inspired by recent advancements in information retrieval using BERT-based architectures. The `pytorch-pretrained-bert` library forms the foundation for BERT tokenization and model loading. Special thanks to the open-source community for datasets and tools supporting IR research.
