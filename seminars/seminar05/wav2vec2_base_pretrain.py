from typing import List, Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch import nn
from torch.nn import functional as F
try:
    from .gumbel_softmax import Wav2Vec2GumbelVectorQuantizer
except ImportError:  # pragma: no cover - allows running this file as a script
    from gumbel_softmax import Wav2Vec2GumbelVectorQuantizer

try:
    from . import components
except ImportError:  # pragma: no cover - allows running this file as a script
    import components

class Wav2Vec2Model(Module):
    """Wav2Vec2 model for both pretraining and fine-tuning.

    When used for pretraining (quantizer provided), implements contrastive learning:
    - Mask spans of feature extractor outputs
    - Predict quantized representations at masked positions
    - Contrastive loss against distractors + diversity loss for codebook usage

    When used for fine-tuning/inference (quantizer=None), provides standard encoder outputs.

    Args:
        feature_extractor (torch.nn.Module): Feature extractor for raw audio.
        encoder (torch.nn.Module): Transformer encoder.
        aux (torch.nn.Module or None, optional): Auxiliary head for fine-tuning.
        quantizer (Wav2Vec2GumbelVectorQuantizer or None, optional): Vector quantizer for pretraining.
        proj_codevector_dim (int or None, optional): Dimension for contrastive projection.
        num_negatives (int): Number of negative samples for contrastive loss.
        contrastive_logits_temperature (float): Temperature for contrastive loss.
        diversity_loss_weight (float): Weight for diversity loss term.
        feature_grad_mult (float or None): Gradient multiplier for feature extractor.
        feat_quantizer_dropout (float): Dropout rate before quantizer.
    """

    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
        quantizer: Optional[Wav2Vec2GumbelVectorQuantizer] = None,
        proj_codevector_dim: Optional[int] = None,
        num_negatives: int = 100,
        contrastive_logits_temperature: float = 0.1,
        diversity_loss_weight: float = 0.1,
        feature_grad_mult: Optional[float] = None,
        feat_quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux
        self.quantizer = quantizer
        self.num_negatives = num_negatives
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.diversity_loss_weight = diversity_loss_weight
        self.feature_grad_mult = feature_grad_mult
        self.feat_quantizer_dropout = feat_quantizer_dropout

        if quantizer is not None and proj_codevector_dim is not None:
            encoder_embed_dim = encoder.feature_projection.projection.out_features
            codevector_dim = quantizer.codevectors.shape[-1] * quantizer.num_groups
            self.project_q = nn.Linear(codevector_dim, proj_codevector_dim)
            self.project_hid = nn.Linear(encoder_embed_dim, proj_codevector_dim)
            self.dropout_features = nn.Dropout(feat_quantizer_dropout)
            # Learned mask embedding vector for pretraining (same dim as encoder output)
            self.masked_spec_embed = nn.Parameter(torch.Tensor(encoder_embed_dim).uniform_())
        else:
            self.project_q = None
            self.project_hid = None
            self.dropout_features = None
            self.masked_spec_embed = None

    def _scale_feature_gradients(self, features: Tensor) -> Tensor:
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0:
            return features * self.feature_grad_mult + features.detach() * (1 - self.feature_grad_mult)
        return features

    def _mask_hidden_states(
        self,
        hidden_states: Tensor,
        mask_time_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Replace masked positions with learned mask embedding.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim).
            mask_time_indices: Boolean tensor of shape (batch, seq_len) where True indicates masked positions.

        Returns:
            Tensor with masked positions replaced by self.masked_spec_embed.
        """
        if mask_time_indices is None or self.masked_spec_embed is None:
            return hidden_states

        # Clone to avoid in-place modification
        hidden_states = hidden_states.clone()
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        return hidden_states

    @staticmethod
    def _get_attention_mask(features: Tensor, lengths: Optional[Tensor]) -> Optional[Tensor]:
        if lengths is None:
            return None
        batch_size, max_len = features.shape[:2]
        return torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]

    @staticmethod
    def _get_feature_vector_attention_mask(
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        output_lengths: Optional[torch.LongTensor] = None,
    ):
        """Compute attention mask for feature vectors from waveform attention mask.

        Args:
            feature_vector_length: Length of feature vector dimension (output of CNN).
            attention_mask: Waveform-level attention mask (1 for valid, 0 for padded).
            output_lengths: Pre-computed feature vector lengths. If None, computed from attention_mask.

        Returns:
            Boolean attention mask for feature vectors (True for valid positions).
        """
        if output_lengths is None:
            # Effectively attention_mask.sum(-1), but not inplace to be able to run
            # on inference mode.
            non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

            # Use the explicit components helper block for proper output math calculation
            output_lengths = components._get_feat_extract_output_lengths(
                non_padded_lengths,
                conv_layers=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
            )

        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        new_attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        new_attention_mask[(torch.arange(batch_size, device=attention_mask.device), output_lengths - 1)] = 1
        new_attention_mask = new_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return new_attention_mask

    def _compute_training_losses(
        self,
        transformer_features: Tensor,
        quantized_features: Tensor,
        codevector_perplexity: Tensor,
        mask_time_indices: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        negative_quantized_features = self._sample_negatives(
            quantized_features, self.num_negatives, attention_mask
        )

        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            self.contrastive_logits_temperature,
        )

        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        preds = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        contrastive_loss = F.cross_entropy(preds.float(), target, reduction="sum")

        num_losses = mask_time_indices.sum()
        contrastive_loss = contrastive_loss / num_losses

        num_codevectors = self.quantizer.num_groups * self.quantizer.num_vars
        diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors
        total_loss = contrastive_loss + self.diversity_loss_weight * diversity_loss
        return total_loss, contrastive_loss, diversity_loss

    def set_gumbel_temperature(self, temperature: float):
        """Set Gumbel-Softmax temperature for quantizer."""
        self.quantizer.set_temperature(temperature)

    def freeze_feature_extractor(self):
        """Disable gradient computation for feature extractor."""
        for param in self.feature_extractor.parameters():
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

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional): Valid lengths. Shape: `(batch, )`.
            num_layers (int or None, optional): Number of intermediate layers.

        Returns:
            (List[Tensor], Optional[Tensor]): Features from layers and valid lengths.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        mask_time_indices: Optional[Tensor] = None,
    ):
        """Forward pass - inference or pretraining mode.

        Args:
            waveforms (Tensor): Audio waveforms. Shape: (batch, frames).
            lengths (Tensor or None, optional): Valid audio lengths. Shape: (batch,).
            mask_time_indices (Tensor or None, optional): If provided, runs pretraining mode.

        Returns:
            If mask_time_indices is None (inference mode):
                Tuple[Tensor, Optional[Tensor]]: Encoder output and lengths.
            If mask_time_indices is provided (pretraining mode):
                Dict[str, Tensor]: Loss dict with contrastive and diversity losses.
        """
        if mask_time_indices is None:
            x, lengths = self.feature_extractor(waveforms, lengths)
            x = self.encoder(x, lengths)
            if self.aux is not None:
                x = self.aux(x)
            return x, lengths

        features, lengths = self.feature_extractor(waveforms, lengths)
        features = self._scale_feature_gradients(features)

        # Build attention mask strictly via _get_feature_vector_attention_mask using wave lengths
        if lengths is not None:
            raw_mask = torch.arange(waveforms.shape[1], device=lengths.device)[None, :] < lengths[:, None]
            attention_mask = self._get_feature_vector_attention_mask(features.shape[1], raw_mask.long())
        else:
            attention_mask = None

        # Project features to encoder dimension (unmasked - for quantizer)
        hidden_states = self.encoder.feature_projection(features)

        # Apply masking: replace masked positions with learned mask embedding
        # Transformer sees MASKED hidden states
        hidden_states_masked = self._mask_hidden_states(hidden_states, mask_time_indices)

        # Build attention mask for transformer (from lengths, not the boolean mask)
        if lengths is not None:
            batch_size, max_len, _ = hidden_states_masked.shape
            attn_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            attn_mask = -10000.0 * attn_mask[:, None, None, :].to(dtype=hidden_states_masked.dtype)
            attn_mask = attn_mask.expand(batch_size, 1, max_len, max_len)
        else:
            attn_mask = None

        # Transformer processes masked hidden states
        encoder_out = self.encoder.transformer(hidden_states_masked, attention_mask=attn_mask)
        transformer_features = self.project_hid(encoder_out)

        # Quantize all (UNMASKED) extracted features
        # IMPORTANT: quantizer receives the ORIGINAL unmasked features, not the masked ones!
        quantized_input = self.dropout_features(features)
        quantized_features, codevector_perplexity = self.quantizer(quantized_input, mask_time_indices)
        quantized_features = self.project_q(quantized_features)

        loss = None
        contrastive_loss_val = None
        diversity_loss_val = None

        if self.training:
            loss, contrastive_loss_val, diversity_loss_val = self._compute_training_losses(
                transformer_features=transformer_features,
                quantized_features=quantized_features,
                codevector_perplexity=codevector_perplexity,
                mask_time_indices=mask_time_indices,
                attention_mask=attention_mask,
            )

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss_val,
            "diversity_loss": diversity_loss_val,
            "projected_states": transformer_features,
            "projected_quantized_states": quantized_features,
            "codevector_perplexity": codevector_perplexity,
        }

    @classmethod
    def create(
        self,
        extractor_mode: str = "group_norm",
        extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]] = None,
        extractor_conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_projection_dropout: float = 0.1,
        encoder_pos_conv_kernel: int = 128,
        encoder_pos_conv_groups: int = 16,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 12,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_features: int = 3072,
        encoder_ff_interm_dropout: float = 0.1,
        encoder_dropout: float = 0.1,
        encoder_layer_norm_first: bool = False,
        encoder_layer_drop: float = 0.1,
        load_pretrained: bool = True,
        dl_kwargs: Optional[dict] = None,
        strict: bool = True,
        codevector_dim: int = 256,
        num_codevector_groups: int = 2,
        num_codevectors_per_group: int = 320,
        proj_codevector_dim: int = 256,
        num_negatives: int = 100,
        contrastive_logits_temperature: float = 0.1,
        diversity_loss_weight: float = 0.1,
        feature_grad_mult: Optional[float] = 0.0,
        feat_quantizer_dropout: float = 0.0,
    ) -> "Wav2Vec2Model":
        """Build Wav2Vec2Model with configurable architecture.

        Args:
            extractor_mode (str): Feature extractor mode ("group_norm" or "layer_norm").
            extractor_conv_layer_config: Conv layer config. None uses default.
            extractor_conv_bias (bool): Whether conv layers have bias.
            encoder_embed_dim (int): Encoder embedding dimension.
            encoder_projection_dropout (float): Dropout after projection.
            encoder_pos_conv_kernel (int): Positional conv kernel size.
            encoder_pos_conv_groups (int): Positional conv groups.
            encoder_num_layers (int): Number of transformer layers.
            encoder_num_heads (int): Number of attention heads.
            encoder_attention_dropout (float): Attention dropout.
            encoder_ff_interm_features (int): Feed-forward intermediate dimension.
            encoder_ff_interm_dropout (float): Feed-forward dropout.
            encoder_dropout (float): Encoder dropout.
            encoder_layer_norm_first (bool): Layer norm before attention.
            encoder_layer_drop (float): Layer dropout probability.
            load_pretrained (bool): Whether to load WAV2VEC2_BASE pretrained weights.
            dl_kwargs (dict or None): Download kwargs for torchaudio.
            strict (bool): Strict loading for state_dict.
            codevector_dim (int): Dimension of codevectors.
            num_codevector_groups (int): Number of codevector groups.
            num_codevectors_per_group (int): Codevectors per group.
            proj_codevector_dim (int): Dimension for contrastive projection.
            num_negatives (int): Number of negative samples.
            contrastive_logits_temperature (float): Contrastive loss temperature.
            diversity_loss_weight (float): Weight for diversity loss.
            feature_grad_mult (float or None): Gradient multiplier for feature extractor.
            feat_quantizer_dropout (float): Dropout before quantizer.

        Returns:
            Wav2Vec2Model: Model ready for pretraining.
        """
        if extractor_conv_layer_config is None:
            extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

        feature_extractor = components._get_feature_extractor(
            extractor_mode, extractor_conv_layer_config, extractor_conv_bias
        )
        encoder = components._get_encoder(
            in_features=extractor_conv_layer_config[-1][0],
            embed_dim=encoder_embed_dim,
            dropout_input=encoder_projection_dropout,
            pos_conv_kernel=encoder_pos_conv_kernel,
            pos_conv_groups=encoder_pos_conv_groups,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            attention_dropout=encoder_attention_dropout,
            ff_interm_features=encoder_ff_interm_features,
            ff_interm_dropout=encoder_ff_interm_dropout,
            dropout=encoder_dropout,
            layer_norm_first=encoder_layer_norm_first,
            layer_drop=encoder_layer_drop,
        )

        quantizer_input_dim = encoder.feature_projection.projection.in_features
        quantizer = Wav2Vec2GumbelVectorQuantizer(
            input_dim=quantizer_input_dim,
            codevector_dim=codevector_dim,
            num_groups=num_codevector_groups,
            num_vars=num_codevectors_per_group,
        )

        model = self(
            feature_extractor=feature_extractor,
            encoder=encoder,
            aux=None,
            quantizer=quantizer,
            proj_codevector_dim=proj_codevector_dim,
            num_negatives=num_negatives,
            contrastive_logits_temperature=contrastive_logits_temperature,
            diversity_loss_weight=diversity_loss_weight,
            feature_grad_mult=feature_grad_mult,
            feat_quantizer_dropout=feat_quantizer_dropout,
        )

        if load_pretrained:
            try:
                import torchaudio
            except ImportError as err:
                raise RuntimeError("torchaudio is required to load WAV2VEC2_BASE pretrained weights.") from err
            pretrained = torchaudio.pipelines.WAV2VEC2_BASE.get_model(dl_kwargs=dl_kwargs)
            model.load_state_dict(pretrained.state_dict(), strict=False)

        return model


def test_wav2vec2_pretrain_forward(
    wav_path: str,
    batch_size: int = 1,
    mask_prob: float = 0.065,
    mask_length: int = 10,
    target_sample_rate: int = 16000,
) -> Dict[str, Tensor]:
    """Test Wav2Vec2PretrainModel forward pass using an input ``.wav`` file.

    Args:
        wav_path (str): Path to input wav file.
        batch_size (int): Number of copies of the input audio in batch.
        mask_prob (float): Probability for mask span start.
        mask_length (int): Length of each masked span.
        target_sample_rate (int): Audio sample rate expected by the model.

    Returns:
        Dict containing model outputs from forward pass.
    """
    try:
        import torchaudio
    except ImportError as err:
        raise RuntimeError("torchaudio is required to load wav files for this test function.") from err

    # Build model and set to training mode
    model = Wav2Vec2Model.create()
    model.train()

    # Load waveform from file (channels, frames)
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # Build batch: (batch, frames)
    waveforms = waveform.squeeze(0).unsqueeze(0).repeat(batch_size, 1)
    audio_lengths = torch.full((batch_size,), waveforms.size(1), dtype=torch.long)

    # Compute feature extractor output length for mask shape
    with torch.no_grad():
        features, _ = model.feature_extractor(waveforms, audio_lengths)
        seq_len = features.shape[1]
    if seq_len < 2:
        raise ValueError(
            f"Input audio is too short after feature extraction (seq_len={seq_len}). "
            "Provide a longer wav file."
        )

    effective_mask_length = min(mask_length, max(1, seq_len - 1))
    if effective_mask_length != mask_length:
        print(
            f"Adjusted mask_length from {mask_length} to {effective_mask_length} "
            f"to match sequence length {seq_len}."
        )

    # Generate mask_time_indices using the utility from components
    mask_time_indices = components._compute_mask_indices(
        shape=(batch_size, seq_len),
        padding_mask=None,
        mask_prob=mask_prob,
        mask_length=effective_mask_length,
        mask_type="static",
    )

    # Run forward pass
    outputs = model(waveforms, lengths=audio_lengths, mask_time_indices=mask_time_indices)

    # Print summary
    print(f"Input wav: {wav_path}")
    print(f"Sample rate: {sample_rate}")
    print(f"Input waveforms shape: {waveforms.shape}")
    print(f"Feature extractor output shape: {features.shape}")
    print(f"Mask shape: {mask_time_indices.shape}")
    print(f"Masked positions: {mask_time_indices.sum().item()}/{mask_time_indices.numel()}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Contrastive loss: {outputs['contrastive_loss'].item():.4f}")
    print(f"Diversity loss: {outputs['diversity_loss'].item():.4f}")
    print(f"Codevector perplexity: {outputs['codevector_perplexity'].item():.2f}")

    return outputs

if __name__ == "__main__":
    out = test_wav2vec2_pretrain_forward(
    wav_path="/Users/oorgien/codes/Dl_audio_repos/DL-Audio-AIMasters-Course/seminars/seminar01/sample1.wav",
    batch_size=1,
    mask_prob=0.05,
    mask_length=10,
)


__all__ = [
    "Wav2Vec2GumbelVectorQuantizer",
    "Wav2Vec2Model",
    "Wav2Vec2PretrainModel",
]

# Backward compatibility alias
Wav2Vec2PretrainModel = Wav2Vec2Model
