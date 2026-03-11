import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """Vector quantization using Gumbel-Softmax.
    
    Discretizes continuous feature vectors using product quantization with
    Gumbel-Softmax relaxation for differentiable training.
    """

    def __init__(
        self,
        input_dim: int,
        codevector_dim: int,
        num_groups: int,  # Theory: G (e.g., 2). Number of sub-codebooks.
        num_vars: int,    # Theory: V (e.g., 320). Number of entries per sub-codebook.
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_vars = num_vars

        # Theory: Product Quantization requires the target dimension to be evenly 
        # split among the G groups. If dim=256 and G=2, each group yields a 128-dim slice.
        assert (
            codevector_dim % num_groups == 0
        ), f"`codevector_dim {codevector_dim} must be divisible by `num_groups` {num_groups}"

        # Codebook: shape (1, G * V, D / G)
        # This stores the actual physical vectors. We store them flattened as one matrix 
        # for efficient batch multiplication later. Total theoretical combinations = V^G.
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, num_groups * num_vars, codevector_dim // num_groups)
        )
        
        # Linear projection to generate logits for each variable in each group.
        # Maps Input_Dim -> G * V. 
        self.weight_proj = nn.Linear(input_dim, num_groups * num_vars)

        # Theory: Tau (τ) controls the sharpness of the Gumbel-Softmax.
        # Usually starts high (e.g., 2.0) for exploration, and is annealed (decayed) 
        # down to a low value (e.g., 0.5 or 0.1) as training progresses to make it closer to true argmax.
        self.temperature = 2.0

        # Uniform initialization breaks symmetry, allowing different codevectors to learn different features.
        nn.init.uniform_(self.codevectors)

    def set_temperature(self, temperature: float):
        """Set the Gumbel-Softmax temperature (used by learning rate schedulers)."""
        self.temperature = temperature

    @staticmethod
    def _compute_perplexity(probs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute perplexity of codebook usage distribution.
        Theory: Max perplexity is V. If Perplexity drops to 1, the model is suffering 
        from severe mode collapse (using only 1 codevector).
        """
        if mask is not None:
            # In Wav2Vec2, loss is often only computed on masked frames.
            # We only want to measure codebook usage on valid/masked time steps.
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            # Marginal probability p_i = (Sum of probabilities for code i) / (Total valid frames)
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            # Average usage of each code across the batch and sequence length
            marginal_probs = probs.mean(dim=0)

        # Theory: Entropy H = -Sum(p * log(p)). Perplexity = exp(H).
        # We add 1e-7 inside the log to prevent log(0) which causes NaN errors.
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(
        self,
        hidden_states: Tensor,
        mask_time_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward Pass Flow:
        1. Map continuous hidden_states -> logits
        2. Convert logits -> probabilities (Gumbel-Softmax/Argmax)
        3. Use probabilities as weights to mix/select the actual codevectors.
        """
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # Step 1: Project continuous vectors to logits for codebook selection.
        # Shape becomes (B, T, G * V)
        hidden_states = self.weight_proj(hidden_states)
        
        # Reshape so we process each group independently.
        # Shape: (B * T * G, V). Every row represents one group's logits to pick from V vars.
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # Step 2 (Training): Sample codevector indices via Gumbel-Softmax.
            # hard=True uses the "Straight-Through Estimator": 
            #   - Forward pass: returns strictly One-Hot vectors (like argmax).
            #   - Backward pass: behaves as if it returned the soft probabilities, allowing gradient flow.
            codevector_probs = F.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # Compute perplexity from soft distribution (without Gumbel noise) 
            # to get an accurate representation of the model's true confidence.
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # Step 2 (Inference): No gradients needed. Standard hard Argmax.
            codevector_idx = hidden_states.argmax(dim=-1)
            
            # Create a one-hot tensor mathematically identical to the forward pass of `hard=True`
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            
            # Reshape for perplexity calculation
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        # Flatten out the groups again. Shape: (B * T, G * V)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)

        # Step 3: Retrieve codevectors using probabilities.
        # Theory: Because codevector_probs is One-Hot, multiplying it with self.codevectors 
        # is mathematically equivalent to an embedding lookup, but it keeps the computation 
        # graph intact so gradients can flow backward to the projection weights.
        
        # Shapes: codevector_probs.unsqueeze(-1) -> (B * T, G * V, 1)
        #         self.codevectors               -> (1,     G * V, D / G)
        # Broadcasting multiplies the 1 or 0 by the specific codevector chunk.
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        
        # Re-arrange and sum to extract the chosen chunks and concatenate them.
        codevectors = (
            # Reshape to (B * T, G, V, D / G)
            codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
            # Sum over V (dimension 2). Since it's one-hot, V-1 of these are zero vectors. 
            # This extracts the exactly chosen vector for each of the G groups.
            # Shape becomes (B * T, G, D / G)
            .sum(-2)
            # View flattens G and D/G. This implicitly concatenates the G chunks together!
            # Shape becomes (B, T, D) -> The final quantized representation!
            .view(batch_size, sequence_length, -1)
        )

        return codevectors, perplexity