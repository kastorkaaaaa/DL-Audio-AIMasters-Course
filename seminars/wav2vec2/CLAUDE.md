# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a wav2vec2/HuBERT/WavLM model implementation from torchaudio, used in a DL Audio AI Masters Course seminar. The code provides PyTorch implementations of self-supervised speech representation models.

## Core Architecture

The codebase implements three related model families that share a common architecture:

- **Wav2Vec2** (`model.py:Wav2Vec2Model`) - Base architecture from the wav2vec 2.0 paper
- **HuBERT** - Uses `Wav2Vec2Model` architecture with different pretraining approach
- **WavLM** - Extends Wav2Vec2 with relative position attention (`wavlm_attention.py:WavLMSelfAttention`)

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
│           ├── attention (SelfAttention or WavLMSelfAttention)
│           └── feed_forward (FeedForward)
└── aux (Optional Linear layer for fine-tuning)
```

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | Model classes and factory functions for all architectures |
| `components.py` | Building blocks: FeatureExtractor, Encoder, SelfAttention, etc. |
| `wavlm_attention.py` | WavLM-specific attention with relative position embeddings |
| `base_model.py` | Simplified wav2vec2-base helpers for seminar use |
| `utils/import_huggingface.py` | Import pretrained weights from HuggingFace |
| `utils/import_fairseq.py` | Import pretrained weights from fairseq |

## Model Factory Functions

All models are built via factory functions in `model.py`:

```python
# Wav2Vec2 variants
wav2vec2_base()       # 12 layers, 768 dim
wav2vec2_large()      # 24 layers, 1024 dim
wav2vec2_large_lv60k() # Large with layer_norm extraction

# HuBERT variants
hubert_base(), hubert_large(), hubert_xlarge()

# WavLM variants
wavlm_base(), wavlm_large()

# XLS-R multilingual variants
wav2vec2_xlsr_300m(), wav2vec2_xlsr_1b(), wav2vec2_xlsr_2b()
```

## Loading Pretrained Weights

```python
# From torchaudio pipelines
import torchaudio
model = wav2vec2_base()
pretrained = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
model.load_state_dict(pretrained.state_dict(), strict=True)

# From HuggingFace
from utils import import_huggingface_model
from transformers import Wav2Vec2ForCTC
hf_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model = import_huggingface_model(hf_model)

# From fairseq (requires fairseq installed)
from utils import import_fairseq_model
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(['model.pt'])
imported = import_fairseq_model(model[0])
```

## Differences Between Architectures

| Feature | Wav2Vec2 Base | Wav2Vec2 Large | HuBERT Large | WavLM |
|---------|---------------|----------------|--------------|-------|
| extractor_mode | group_norm | group_norm | layer_norm | group_norm/base, layer_norm/large |
| conv_bias | False | False | False | False |
| layer_norm_first | False | False | True | False/base, True/large |
| Relative position | No | No | No | Yes |

## Audio Input

- Sample rate: 16kHz (use `WAV2VEC2_BASE_SAMPLE_RATE` from `base_model.py`)
- Input shape: `(batch, frames)` - raw waveform
- Feature extraction downsamples by ~320x (7 conv layers with strides)
