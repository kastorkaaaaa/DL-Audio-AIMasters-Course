"""Minimal wav2vec2-base-only helpers for seminar code.

This module intentionally keeps only wav2vec2 base builders so students
do not need to navigate unrelated model variants.
"""

from typing import Dict, Mapping, Optional, Tuple

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F

from model import Wav2Vec2Model, wav2vec2_base as _wav2vec2_base_local

WAV2VEC2_BASE_SAMPLE_RATE = torchaudio.pipelines.WAV2VEC2_BASE.sample_rate


def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds wav2vec2 base architecture only."""
    return _wav2vec2_base_local(
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def wav2vec2_base_pretrained(
    dl_kwargs: Optional[Mapping[str, object]] = None,
    strict: bool = True,
) -> Wav2Vec2Model:
    """Builds wav2vec2 base and loads ``WAV2VEC2_BASE`` pretrained weights."""
    model = wav2vec2_base()
    download_args = dict(dl_kwargs) if dl_kwargs is not None else None
    pretrained = torchaudio.pipelines.WAV2VEC2_BASE.get_model(dl_kwargs=download_args)
    model.load_state_dict(pretrained.state_dict(), strict=strict)
    return model


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


class Wav2Vec2GumbelVectorQuantizer(nn.Module):                                                                                                         
    """Vector quantization using Gumbel-Softmax.                                                                                                        
                                                                                                                                                        
    Discretizes continuous feature vectors using product quantization with                                                                              
    Gumbel-Softmax relaxation for differentiable training.                                                                                              
                                                                                                                                                        
    Reference: `Categorical Reparameterization with Gumbel-Softmax                                                                                      
    <https://arxiv.org/pdf/1611.01144.pdf>`__                                                                                                           
                                                                                                                                                        
    Args:                                                                                                                                               
        input_dim (int): Dimension of input features (from feature extractor).                                                                          
        codevector_dim (int): Dimension of output codevectors.                                                                                          
        num_groups (int): Number of codevector groups (V).                                                                                              
        num_vars (int): Number of codevectors per group (K).                                                                                            
    """                                                                                                                                                 
                                                                                                                                                        
    def __init__(                                                                                                                                       
        self,                                                                                                                                           
        input_dim: int,                                                                                                                                 
        codevector_dim: int,                                                                                                                            
        num_groups: int,                                                                                                                                
        num_vars: int,                                                                                                                                  
    ):                                                                                                                                                  
        super().__init__()                                                                                                                              
        self.num_groups = num_groups                                                                                                                    
        self.num_vars = num_vars                                                                                                                        
                                                                                                                                                        
        assert (                                                                                                                                        
            codevector_dim % num_groups == 0                                                                                                            
        ), f"`codevector_dim {codevector_dim} must be divisible by `num_groups` {num_groups}"                                                           
                                                                                                                                                        
        # Codebook: shape (1, num_groups * num_vars, codevector_dim // num_groups)                                                                      
        self.codevectors = nn.Parameter(                                                                                                                
            torch.FloatTensor(1, num_groups * num_vars, codevector_dim // num_groups)                                                                   
        )                                                                                                                                               
        self.weight_proj = nn.Linear(input_dim, num_groups * num_vars)                                                                                  
                                                                                                                                                        
        # Temperature for Gumbel-Softmax (can be decayed during training)                                                                               
        self.temperature = 2.0                                                                                                                          
                                                                                                                                                        
        # Initialize codevectors                                                                                                                        
        nn.init.uniform_(self.codevectors)                                                                                                              
                                                                                                                                                        
    def set_temperature(self, temperature: float):                                                                                                      
        """Set the Gumbel-Softmax temperature."""                                                                                                       
        self.temperature = temperature                                                                                                                  
                                                                                                                                                        
    @staticmethod                                                                                                                                       
    def _compute_perplexity(probs: Tensor, mask: Optional[Tensor] = None) -> Tensor:                                                                    
        """Compute perplexity of codebook usage distribution.                                                                                           
                                                                                                                                                        
        Higher perplexity indicates more uniform codebook usage.                                                                                        
        """                                                                                                                                             
        if mask is not None:                                                                                                                            
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)                                                                           
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))                                                                          
            marginal_probs = probs.sum(dim=0) / mask.sum()                                                                                              
        else:                                                                                                                                           
            marginal_probs = probs.mean(dim=0)                                                                                                          
                                                                                                                                                        
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()                                             
        return perplexity                                                                                                                               
                                                                                                                                                        
    def forward(                                                                                                                                        
        self,                                                                                                                                           
        hidden_states: Tensor,                                                                                                                          
        mask_time_indices: Optional[Tensor] = None,                                                                                                     
    ) -> Tuple[Tensor, Tensor]:                                                                                                                         
        """Quantize hidden states using Gumbel-Softmax.                                                                                                 
                                                                                                                                                        
        Args:                                                                                                                                           
            hidden_states (Tensor): Features from feature extractor. Shape: (batch, seq_len, input_dim).                                                
            mask_time_indices (Tensor or None): Boolean mask for valid positions. Shape: (batch, seq_len).                                              
                                                                                                                                                        
        Returns:                                                                                                                                        
            codevectors (Tensor): Quantized representations. Shape: (batch, seq_len, codevector_dim).                                                   
            perplexity (Tensor): Codebook usage perplexity (scalar).                                                                                    
        """                                                                                                                                             
        batch_size, sequence_length, hidden_size = hidden_states.shape                                                                                  
                                                                                                                                                        
        # Project to codevector logits                                                                                                                  
        hidden_states = self.weight_proj(hidden_states)                                                                                                 
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)                                                          
                                                                                                                                                        
        if self.training:                                                                                                                               
            # Sample codevector indices via Gumbel-Softmax (differentiable)                                                                             
            codevector_probs = F.gumbel_softmax(                                                                                                        
                hidden_states.float(), tau=self.temperature, hard=True                                                                                  
            ).type_as(hidden_states)                                                                                                                    
                                                                                                                                                        
            # Compute perplexity from soft distribution                                                                                                 
            codevector_soft_dist = torch.softmax(                                                                                                       
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1                                                   
            )                                                                                                                                           
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)                                                              
        else:                                                                                                                                           
            # Take argmax in non-differentiable way (inference)                                                                                         
            codevector_idx = hidden_states.argmax(dim=-1)                                                                                               
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(                                                                  
                -1, codevector_idx.view(-1, 1), 1.0                                                                                                     
            )                                                                                                                                           
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)                                                 
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)                                                                  
                                                                                                                                                        
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)                                                                      
                                                                                                                                                        
        # Retrieve codevectors using probabilities                                                                                                      
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors                                                                       
        codevectors = (                                                                                                                                 
            codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)                                                
            .sum(-2)                                                                                                                                    
            .view(batch_size, sequence_length, -1)                                                                                                      
        )                                                                                                                                               
                                                                                                                                                        
        return codevectors, perplexity 


class Wav2Vec2PretrainModel(nn.Module):
    """Wav2Vec2 model for self-supervised pretraining.

    Implements the contrastive learning objective from wav2vec 2.0:
    - Mask spans of feature extractor outputs
    - Predict quantized representations at masked positions
    - Contrastive loss against distractors + diversity loss for codebook usage

    Args:
        wav2vec2 (Wav2Vec2Model): Base wav2vec2 model.
        quantizer (Wav2Vec2GumbelVectorQuantizer): Vector quantizer module.
        proj_codevector_dim (int): Dimension for contrastive loss projection.
        num_negatives (int): Number of negative samples for contrastive loss.
        contrastive_logits_temperature (float): Temperature for contrastive loss.
        diversity_loss_weight (float): Weight for diversity loss term.
        feature_grad_mult (float or None): Gradient multiplier for feature extractor.
        feat_quantizer_dropout (float): Dropout rate before quantizer.
    """

    def __init__(
        self,
        wav2vec2: Wav2Vec2Model,
        quantizer: Wav2Vec2GumbelVectorQuantizer,
        proj_codevector_dim: int,
        num_negatives: int = 100,
        contrastive_logits_temperature: float = 0.1,
        diversity_loss_weight: float = 0.1,
        feature_grad_mult: Optional[float] = None,
        feat_quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.quantizer = quantizer
        self.num_negatives = num_negatives
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.diversity_loss_weight = diversity_loss_weight
        self.feature_grad_mult = feature_grad_mult
        self.feat_quantizer_dropout = feat_quantizer_dropout

        # Projection layers for contrastive learning
        encoder_embed_dim = wav2vec2.encoder.feature_projection.projection.out_features
        codevector_dim = quantizer.codevectors.shape[-1] * quantizer.num_groups

        self.project_q = nn.Linear(codevector_dim, proj_codevector_dim)
        self.project_hid = nn.Linear(encoder_embed_dim, proj_codevector_dim)
        self.dropout_features = nn.Dropout(feat_quantizer_dropout)

    def set_gumbel_temperature(self, temperature: float):
        """Set Gumbel-Softmax temperature for quantizer."""
        self.quantizer.set_temperature(temperature)

    def freeze_feature_extractor(self):
        """Disable gradient computation for feature extractor."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    @staticmethod
    def _sample_negatives(
        features: Tensor,
        num_negatives: int,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample negative examples for contrastive loss.

        Args:
            features (Tensor): Features to sample from. Shape: (batch, seq_len, hidden_dim).
            num_negatives (int): Number of negatives per position.
            attention_mask (Tensor or None): Valid position mask. Shape: (batch, seq_len).

        Returns:
            Tensor: Sampled negatives. Shape: (num_negatives, batch, seq_len, hidden_dim).
        """
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape "
                f"(batch_size, sequence_length, hidden_size) = ({batch_size}, {sequence_length}, {hidden_size})."
            )

        features = features.view(-1, hidden_size)  # (batch * seq_len, hidden_dim)

        with torch.no_grad():
            sampled_negative_indices = []
            for batch_idx in range(batch_size):
                high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
                sampled_indices_slice = torch.randint(
                    0, high, size=(num_negatives * sequence_length,), device=features.device
                )
                sampled_negative_indices.append(sampled_indices_slice)

            sampled_negative_indices = torch.stack(sampled_negative_indices)

            # Generate indices of positive vectors
            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                .expand(sequence_length, num_negatives)
                .flatten()
            )

            # Avoid sampling the same positive vector
            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        # Correct for batch size
        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        # Gather negative vectors
        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(
            batch_size, sequence_length, num_negatives, hidden_size
        ).permute(2, 0, 1, 3)

        return sampled_negatives

    @staticmethod
    def compute_contrastive_logits(
        target_features: Tensor,
        negative_features: Tensor,
        predicted_features: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Compute logits for contrastive loss using cosine similarity.

        Args:
            target_features (Tensor): Positive targets. Shape: (batch, seq_len, proj_dim).
            negative_features (Tensor): Negatives. Shape: (num_neg, batch, seq_len, proj_dim).
            predicted_features (Tensor): Predictions from context encoder. Shape: (batch, seq_len, proj_dim).
            temperature (float): Temperature scaling factor.

        Returns:
            Tensor: Contrastive logits. Shape: (num_neg + 1, batch, seq_len).
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(
            predicted_features.float(), target_features.float(), dim=-1
        ).type_as(target_features)

        logits = logits / temperature
        return logits

    def forward(
        self,
        waveforms: Tensor,
        mask_time_indices: Tensor,
        audio_lengths: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute wav2vec2 pretraining loss.

        Args:
            waveforms (Tensor): Audio waveforms. Shape: (batch, frames).
            mask_time_indices (Tensor): Boolean mask indicating masked positions.
                Shape: (batch, seq_len) where seq_len is feature extractor output length.
            audio_lengths (Tensor or None): Valid audio lengths. Shape: (batch,).

        Returns:
            Dict containing:
                - loss (Tensor): Total loss (contrastive + diversity).
                - contrastive_loss (Tensor): Contrastive loss component.
                - diversity_loss (Tensor): Diversity loss component.
                - projected_states (Tensor): Projected encoder outputs.
                - projected_quantized_states (Tensor): Projected quantized targets.
                - codevector_perplexity (Tensor): Codebook usage perplexity.
        """
        # Extract features
        features, lengths = self.wav2vec2.feature_extractor(waveforms, audio_lengths)

        # Apply gradient multiplier for feature extractor
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0:
            features = features * self.feature_grad_mult + features.detach() * (1 - self.feature_grad_mult)

        # Compute attention mask from lengths if provided
        attention_mask = None
        if lengths is not None:
            batch_size, max_len = features.shape[:2]
            attention_mask = (
                torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]
            )

        # Encode with masking (mask applied in transformer via attention)
        # Note: mask_time_indices should mask encoder input, not attention
        encoder_out = self.wav2vec2.encoder(features, lengths)
        transformer_features = self.project_hid(encoder_out)

        # Quantize features (targets)
        quantized_input = self.dropout_features(features)
        quantized_features, codevector_perplexity = self.quantizer(quantized_input, mask_time_indices)
        quantized_features = self.project_q(quantized_features)

        loss = None
        contrastive_loss_val = None
        diversity_loss_val = None

        if self.training:
            # Sample negatives for contrastive loss
            negative_quantized_features = self._sample_negatives(
                quantized_features, self.num_negatives, attention_mask
            )

            # Compute contrastive logits
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.contrastive_logits_temperature,
            )

            # Mask out negatives identical to positives (low codebook utilization)
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # Compute contrastive loss
            preds = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
            contrastive_loss_val = F.cross_entropy(preds.float(), target, reduction="sum")

            # Compute diversity loss
            num_codevectors = self.quantizer.num_groups * self.quantizer.num_vars
            diversity_loss_val = (num_codevectors - codevector_perplexity) / num_codevectors

            # Total loss
            loss = contrastive_loss_val + self.diversity_loss_weight * diversity_loss_val

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss_val,
            "diversity_loss": diversity_loss_val,
            "projected_states": transformer_features,
            "projected_quantized_states": quantized_features,
            "codevector_perplexity": codevector_perplexity,
        }


def wav2vec2_base_pretrain(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    # Quantizer config
    codevector_dim: int = 256,
    num_codevector_groups: int = 2,
    num_codevectors_per_group: int = 320,
    # Pretraining config
    proj_codevector_dim: int = 256,
    num_negatives: int = 100,
    contrastive_logits_temperature: float = 0.1,
    diversity_loss_weight: float = 0.1,
    feature_grad_mult: Optional[float] = 0.0,
    feat_quantizer_dropout: float = 0.0,
) -> Wav2Vec2PretrainModel:
    """Builds Wav2Vec2PretrainModel with base architecture.

    Args:
        encoder_projection_dropout (float): Dropout after feature projection.
        encoder_attention_dropout (float): Dropout in attention layers.
        encoder_ff_interm_dropout (float): Dropout in feed-forward layers.
        encoder_dropout (float): Dropout at end of encoder layers.
        encoder_layer_drop (float): Probability of dropping encoder layers.
        codevector_dim (int): Dimension of codevectors.
        num_codevector_groups (int): Number of codevector groups (V).
        num_codevectors_per_group (int): Codevectors per group (K).
        proj_codevector_dim (int): Dimension for contrastive projection.
        num_negatives (int): Number of negative samples for contrastive loss.
        contrastive_logits_temperature (float): Temperature for contrastive loss.
        diversity_loss_weight (float): Weight for diversity loss.
        feature_grad_mult (float or None): Gradient multiplier for feature extractor.
        feat_quantizer_dropout (float): Dropout before quantizer.

    Returns:
        Wav2Vec2PretrainModel: Model ready for pretraining.
    """
    # Build base wav2vec2 (no aux head needed for pretraining)
    wav2vec2 = wav2vec2_base_pretrained()

    # Build quantizer (input_dim = feature extractor output = 512 for base)
    quantizer = Wav2Vec2GumbelVectorQuantizer(
        input_dim=512,  # Feature extractor output channels
        codevector_dim=codevector_dim,
        num_groups=num_codevector_groups,
        num_vars=num_codevectors_per_group,
    )

    # Build pretrain model
    model = Wav2Vec2PretrainModel(
        wav2vec2=wav2vec2,
        quantizer=quantizer,
        proj_codevector_dim=proj_codevector_dim,
        num_negatives=num_negatives,
        contrastive_logits_temperature=contrastive_logits_temperature,
        diversity_loss_weight=diversity_loss_weight,
        feature_grad_mult=feature_grad_mult,
        feat_quantizer_dropout=feat_quantizer_dropout,
    )

    return model


__all__ = [
    "WAV2VEC2_BASE_SAMPLE_RATE",
    "Wav2Vec2GumbelVectorQuantizer",
    "Wav2Vec2PretrainModel",
    "compute_wav2vec2_full_loss",
    "wav2vec2_base",
    "wav2vec2_base_pretrained",
    "wav2vec2_base_pretrain",
]

if __name__ == "__main__":
    model = wav2vec2_base_pretrain()


