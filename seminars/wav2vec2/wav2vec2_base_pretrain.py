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
    """Acoustic model used in *wav2vec 2.0* :cite:`baevski2020wav2vec`.

    Note:
        To build the model, please use one of the factory functions.

    See Also:
        * :class:`torchaudio.pipelines.Wav2Vec2Bundle`: Pretrained models (without fine-tuning)
        * :class:`torchaudio.pipelines.Wav2Vec2ASRBundle`: ASR pipelines with pretrained models.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """  # noqa: E501

    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        This returns the list of outputs from the intermediate layers of
        transformer block in encoder.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that the entire audio waveform
                length is valid.
            num_layers (int or None, optional):
                If given, limit the number of intermediate layers to go through.
                Providing `1` will stop the computation after going through one
                intermediate layers. If not given, the outputs from all the
                intermediate layers are returned.

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                Features from requested layers.
                Each Tensor is of shape: `(batch, time frame, feature dimension)`
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of each feature Tensor.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths
    

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

    def _scale_feature_gradients(self, features: Tensor) -> Tensor:
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0:
            return features * self.feature_grad_mult + features.detach() * (1 - self.feature_grad_mult)
        return features

    @staticmethod
    def _get_attention_mask(features: Tensor, lengths: Optional[Tensor]) -> Optional[Tensor]:
        if lengths is None:
            return None
        batch_size, max_len = features.shape[:2]
        return torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]

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

        num_codevectors = self.quantizer.num_groups * self.quantizer.num_vars
        diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors
        total_loss = contrastive_loss + self.diversity_loss_weight * diversity_loss
        return total_loss, contrastive_loss, diversity_loss

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
        features, lengths = self.wav2vec2.feature_extractor(waveforms, audio_lengths)
        features = self._scale_feature_gradients(features)
        attention_mask = self._get_attention_mask(features, lengths)

        encoder_out = self.wav2vec2.encoder(features, lengths)
        transformer_features = self.project_hid(encoder_out)

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
    

def wav2vec2_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
) -> Wav2Vec2Model:
    """Builds custom :class:`~torchaudio.models.Wav2Vec2Model`.

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        :cite:`baevski2020wav2vec` paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.
        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
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
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, encoder, aux)


def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds "base" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def wav2vec2_base_pretrained(
    dl_kwargs: Optional[dict] = None,
    strict: bool = True,
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
) -> Wav2Vec2Model:
    """Builds :func:`wav2vec2_base` and loads ``WAV2VEC2_BASE`` pretrained weights.

    Args:
        dl_kwargs (dict or None, optional):
            Passed to ``torchaudio.pipelines.WAV2VEC2_BASE.get_model`` for
            checkpoint download behavior.
        strict (bool, optional):
            Passed to ``load_state_dict``. Default: ``True``.
        encoder_projection_dropout (float):
            Passed to :py:func:`wav2vec2_base`.
        encoder_attention_dropout (float):
            Passed to :py:func:`wav2vec2_base`.
        encoder_ff_interm_dropout (float):
            Passed to :py:func:`wav2vec2_base`.
        encoder_dropout (float):
            Passed to :py:func:`wav2vec2_base`.
        encoder_layer_drop (float):
            Passed to :py:func:`wav2vec2_base`.

    Returns:
        Wav2Vec2Model:
            The resulting model with pretrained weights loaded.
    """
    try:
        import torchaudio
    except ImportError as err:
        raise RuntimeError("torchaudio is required to load WAV2VEC2_BASE pretrained weights.") from err

    model = wav2vec2_base(
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_drop=encoder_layer_drop,
    )
    pretrained = torchaudio.pipelines.WAV2VEC2_BASE.get_model(dl_kwargs=dl_kwargs)
    model.load_state_dict(pretrained.state_dict(), strict=strict)
    return model



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

    wav2vec2 = wav2vec2_base_pretrained(
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_drop=encoder_layer_drop,
    )
    quantizer_input_dim = wav2vec2.encoder.feature_projection.projection.in_features

    quantizer = Wav2Vec2GumbelVectorQuantizer(
        input_dim=quantizer_input_dim,
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
    model = wav2vec2_base_pretrain()
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
        features, _ = model.wav2vec2.feature_extractor(waveforms, audio_lengths)
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
    outputs = model(waveforms, mask_time_indices, audio_lengths)

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
    mask_prob=0.01,
    mask_length=2,
)


__all__ = [
    "Wav2Vec2GumbelVectorQuantizer",
    "Wav2Vec2PretrainModel",
    "Wav2Vec2Model",
    "wav2vec2_model",
    "wav2vec2_base",
    "wav2vec2_base_pretrained",
    "wav2vec2_base_pretrain",
]
