# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a wav2vec2/HuBERT model implementation from torchaudio, used in a DL Audio AI Masters Course seminar. The code provides PyTorch implementations of self-supervised speech representation models.

## Core Architecture

The codebase implements two related model families that share a common architecture:

- **Wav2Vec2** (`wav2vec2_base_pretrain.py:Wav2Vec2Model`) - Base architecture from wav2vec 2.0 paper
- **HuBERT** (`model.py:HuBERTPretrainModel`) - Uses `Wav2Vec2Model` architecture with masked prediction pretraining

### Model Components

```
Wav2Vec2Model
├── feature_extractor (FeatureExtractor)
│   └── 7 ConvLayerBlocks that downsample audio
├── encoder (Encoder)
│   ├── feature_projection (FeatureProjection)
│   └── transformer (Transformer)
│       ├── pos_conv_embed (ConvolutionalPositionalEmbedding)
│       └── layers (ModuleList of EncoderLayer)
│           ├── attention (SelfAttention)
│           └── feed_forward (FeedForward)
└── aux (Optional Linear layer for fine-tuning)
```

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | HuBERT pretraining model and factory functions for all architectures |
| `components.py` | Building blocks: FeatureExtractor, Encoder, SelfAttention, MaskGenerator, etc. |
| `wav2vec2_base_pretrain.py` | Wav2Vec2 pretraining model with contrastive learning objective |
| `gumbel_softmax.py` | Vector quantization using Gumbel-Softmax for wav2vec2 pretraining |
| `base_model.py` | CTC loss computation for wav2vec2 fine-tuning |

## Model Factory Functions

### Wav2Vec2 Variants (from `model.py`)

```python
wav2vec2_large()        # 24 layers, 1024 dim
wav2vec2_large_lv60k()  # Large with layer_norm extraction
```

### Wav2Vec2 Pretraining (from `wav2vec2_base_pretrain.py`)

```python
wav2vec2_base_pretrain()  # Base model with contrastive loss + quantizer
```

### HuBERT Variants (from `model.py`)

```python
# Fine-tuning
hubert_base()    # 12 layers, 768 dim
hubert_large()   # 24 layers, 1024 dim
hubert_xlarge()  # 48 layers, 1280 dim

# Pretraining
hubert_pretrain_base()    # Base with mask prediction
hubert_pretrain_large()   # Large with mask prediction
hubert_pretrain_xlarge()  # Extra large with mask prediction
```

## Pretraining Objectives

### Wav2Vec2 Pretraining (`Wav2Vec2PretrainModel`)

Implements contrastive learning from wav2vec 2.0 paper:

1. **Feature Extraction**: Raw audio → CNN feature extractor → feature vectors
2. **Masking**: Mask spans of feature vectors (not implemented in current forward pass)
3. **Context Encoding**: Transformer encoder processes (masked) features
4. **Quantization**: Gumbel-Softmax vector quantization of original features
5. **Contrastive Loss**: Predict quantized representation at masked positions

Key components:
- `Wav2Vec2GumbelVectorQuantizer` (`gumbel_softmax.py`) - Product quantization with G groups, V entries each
- Temperature annealing from 2.0 (exploration) to 0.1 (exploitation)
- Diversity loss to encourage codebook usage

### HuBERT Pretraining (`HuBERTPretrainModel`)

Implements masked prediction from HuBERT paper:

1. **Feature Extraction**: Raw audio → CNN feature extractor → feature vectors
2. **Masking**: `MaskGenerator` masks ~80% of spans (length 10)
3. **Context Encoding**: Transformer encoder processes masked features
4. **Classification**: `LogitGenerator` predicts cluster assignments

## Fine-tuning for ASR

Use `compute_wav2vec2_full_loss` from `base_model.py`:

```python
from base_model import compute_wav2vec2_full_loss, WAV2VEC2_BASE_SAMPLE_RATE

# Model must have aux head for classification
model = hubert_base(aux_num_out=vocab_size)  # or wav2vec2 variants

# Compute CTC loss
result = compute_wav2vec2_full_loss(
    model=model,
    waveforms=audio_batch,          # (batch, frames) at 16kHz
    targets=target_tokens,          # flattened or (batch, max_len)
    target_lengths=target_lengths,  # (batch,)
    waveforms_lengths=audio_lengths,  # optional, (batch,)
    blank_id=0,
    ctc_weight=1.0,
    feature_penalty_weight=0.0,     # L2 penalty on features
)
loss = result["loss"]
```

## Architecture Differences

| Feature | Wav2Vec2 Base | Wav2Vec2 Large | HuBERT Base | HuBERT Large |
|---------|---------------|----------------|-------------|--------------|
| extractor_mode | group_norm | group_norm | group_norm | layer_norm |
| conv_bias | False | False | False | False |
| layer_norm_first | False | False | False | True |
| encoder_embed_dim | 768 | 1024 | 768 | 1024 |
| encoder_layers | 12 | 24 | 12 | 24 |
| encoder_heads | 12 | 16 | 12 | 16 |
| ff_interm_dim | 3072 | 4096 | 3072 | 4096 |

## Audio Input

- Sample rate: 16kHz (use `WAV2VEC2_BASE_SAMPLE_RATE` from `base_model.py`)
- Input shape: `(batch, frames)` - raw waveform
- Feature extraction downsamples by ~320x (7 conv layers with strides 5,2,2,2,2,2,2)

## Vector Quantization Details (`gumbel_softmax.py`)

```python
quantizer = Wav2Vec2GumbelVectorQuantizer(
    input_dim=512,        # Feature extractor output channels
    codevector_dim=256,   # Dimension per codevector
    num_groups=2,         # G groups (product quantization)
    num_vars=320,         # V entries per group
)
# Total combinations: 320^2 = 102,400 unique codevectors
```

Training: Gumbel-Softmax with straight-through estimator
Inference: Hard argmax

## Building Blocks (`components.py`)

### Feature Extraction
- `ConvLayerBlock` - Single conv layer with optional GroupNorm/LayerNorm + GELU
- `FeatureExtractor` - Stack of 7 `ConvLayerBlock`s
- `FeatureProjection` - LayerNorm + Linear projection to encoder dimension

### Encoder
- `ConvolutionalPositionalEmbedding` - Depthwise conv for relative position
- `SelfAttention` - Multi-head self-attention with scaled dot-product
- `FeedForward` - Two linear layers with GELU activation
- `EncoderLayer` - Attention + FFN with residual connections
- `Transformer` - Stack of `EncoderLayer`s with layer dropout
- `Encoder` - Feature projection + Transformer

### Pretraining Utilities
- `MaskGenerator` - Generate span masks for masked prediction
- `LogitGenerator` - Project and compute logits for classification
- `GradMultiply` - Gradient scaling for feature extractor
- `_compute_mask_indices` - Low-level mask computation

## Code Navigation

- `Wav2Vec2Model`: `wav2vec2_base_pretrain.py:13`
- `Wav2Vec2PretrainModel`: `wav2vec2_base_pretrain.py:125`
- `HuBERTPretrainModel`: `model.py:17`
- `Wav2Vec2GumbelVectorQuantizer`: `gumbel_softmax.py:7`
- `compute_wav2vec2_full_loss`: `base_model.py:17`
- `FeatureExtractor`: `components.py:102`
- `Encoder`: `components.py:466`
- `SelfAttention`: `components.py:237`
- `MaskGenerator`: `components.py:972`
