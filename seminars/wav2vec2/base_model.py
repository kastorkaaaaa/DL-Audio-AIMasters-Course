"""Minimal wav2vec2-base-only helpers for seminar code.

This module intentionally keeps only wav2vec2 base builders so students
do not need to navigate unrelated model variants.
"""

from typing import Dict, Mapping, Optional, Tuple

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F

WAV2VEC2_BASE_SAMPLE_RATE = torchaudio.pipelines.WAV2VEC2_BASE.sample_rate


def compute_wav2vec2_full_loss(
    model: Wav2Vec2Model,
    waveforms: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    waveforms_lengths: Optional[Tensor] = None,
    blank_id: int = 0,
    ctc_reduction: str = "mean",
    zero_infinity: bool = True,
    ctc_weight: float = 1.0,
    feature_penalty_weight: float = 0.0,
) -> Dict[str, Tensor]:
    """Computes wav2vec2 fine-tuning loss (CTC + optional feature penalty).

    Notes:
        - The model must have ``aux`` head (set ``aux_num_out`` when building).
        - ``targets`` can be shape ``(batch, max_target_len)`` or flattened 1-D.
    """
    if model.aux is None:
        raise ValueError(
            "CTC loss requires classification head. Build model with aux_num_out (vocab size), "
            "e.g. wav2vec2_base(aux_num_out=vocab_size)."
        )

    features, output_lengths = model.feature_extractor(waveforms, waveforms_lengths)
    feature_penalty = features.float().pow(2).mean()

    emissions = model.encoder(features, output_lengths)
    emissions = model.aux(emissions)
    log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)  # (time, batch, classes)

    if output_lengths is None:
        input_lengths = torch.full(
            size=(waveforms.size(0),),
            fill_value=emissions.size(1),
            dtype=torch.long,
            device=emissions.device,
        )
    else:
        input_lengths = output_lengths.to(dtype=torch.long)

    ctc_loss = F.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths.to(dtype=torch.long),
        blank=blank_id,
        reduction=ctc_reduction,
        zero_infinity=zero_infinity,
    )
    total_loss = ctc_weight * ctc_loss + feature_penalty_weight * feature_penalty
    return {
        "loss": total_loss,
        "ctc_loss": ctc_loss,
        "feature_penalty": feature_penalty,
        "emissions": emissions,
        "output_lengths": input_lengths,
    }



