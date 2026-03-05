"""Commonly used neural network building blocks for BatDetect2 models.

This module provides various reusable `torch.nn.Module` subclasses that form
the fundamental building blocks for constructing convolutional neural network
architectures, particularly encoder-decoder backbones used in BatDetect2.

It includes standard components like basic convolutional blocks (`ConvBlock`),
blocks incorporating downsampling (`StandardConvDownBlock`), and blocks with
upsampling (`StandardConvUpBlock`).

Additionally, it features specialized layers investigated in BatDetect2
research:

- `SelfAttention`: Applies self-attention along the time dimension, enabling
  the model to weigh information across the entire temporal context, often
  used in the bottleneck of an encoder-decoder.
- `FreqCoordConvDownBlock` / `FreqCoordConvUpBlock`: Implement the "CoordConv"
   concept by concatenating normalized frequency coordinate information as an
   extra channel to the input of convolutional layers. This explicitly provides
   spatial frequency information to filters, potentially enabling them to learn
   frequency-dependent patterns more effectively.

These blocks can be utilized directly in custom PyTorch model definitions or
assembled into larger architectures.

A unified factory function `build_layer_from_config` allows creating instances
of these blocks based on configuration objects.
"""

from typing import Annotated, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from batdetect2.core.configs import BaseConfig

__all__ = [
    "ConvBlock",
    "LayerGroupConfig",
    "VerticalConv",
    "FreqCoordConvDownBlock",
    "FreqCoordConvDownBroadcastBlock",
    "StandardConvDownBlock",
    "FreqCoordConvUpBlock",
    "FreqCoordConvUpBroadcastBlock",
    "StandardConvUpBlock",
    "SelfAttention",
    "MultiHeadAttention",
    "LiteMLA",
    "PhiNetConvBlock",
    "VectorQuantizer",
    "VariationalVectorQuantizer",
    "XiConv",
    "XiConvUpBlock",
    "ConvConfig",
    "FreqCoordConvDownConfig",
    "FreqCoordConvDownBroadcastConfig",
    "StandardConvDownConfig",
    "FreqCoordConvUpConfig",
    "FreqCoordConvUpBroadcastConfig",
    "StandardConvUpConfig",
    "VectorQuantizerConfig",
    "VariationalVectorQuantizerConfig",
    "XiConvConfig",
    "XiConvDownConfig",
    "XiConvUpConfig",
    "LayerConfig",
    "build_layer_from_config",
]


class SelfAttentionConfig(BaseConfig):
    name: Literal["SelfAttention"] = "SelfAttention"
    attention_channels: int
    temperature: float = 1


class MultiHeadAttentionConfig(BaseConfig):
    name: Literal["MultiHeadAttention"] = "MultiHeadAttention"
    attention_channels: int
    num_heads: int = 4
    dropout: float = 0.0
    temperature: float = 1.0


class SelfAttention(nn.Module):
    """Self-Attention mechanism operating along the time dimension.

    This module implements a scaled dot-product self-attention mechanism,
    specifically designed here to operate across the time steps of an input
    feature map, typically after spatial dimensions (like frequency) have been
    condensed or squeezed.

    By calculating attention weights between all pairs of time steps, it allows
    the model to capture long-range temporal dependencies and focus on relevant
    parts of the sequence. It's often employed in the bottleneck or
    intermediate layers of an encoder-decoder architecture to integrate global
    temporal context.

    The implementation uses linear projections to create query, key, and value
    representations, computes scaled dot-product attention scores, applies
    softmax, and produces an output by weighting the values according to the
    attention scores, followed by a final linear projection. Positional encoding
    is not explicitly included in this block.

    Parameters
    ----------
    in_channels : int
        Number of input channels (features per time step after spatial squeeze).
    attention_channels : int
        Number of channels for the query, key, and value projections. Also the
        dimension of the output projection's input.
    temperature : float, default=1.0
        Scaling factor applied *before* the final projection layer. Can be used
        to adjust the sharpness or focus of the attention mechanism, although
        scaling within the softmax (dividing by sqrt(dim)) is more common for
        standard transformers. Here it scales the weighted values.

    Attributes
    ----------
    key_fun : nn.Linear
        Linear layer for key projection.
    value_fun : nn.Linear
        Linear layer for value projection.
    query_fun : nn.Linear
        Linear layer for query projection.
    pro_fun : nn.Linear
        Final linear projection layer applied after attention weighting.
    temperature : float
        Scaling factor applied before final projection.
    att_dim : int
        Dimensionality of the attention space (`attention_channels`).
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int,
        temperature: float = 1.0,
    ):
        super().__init__()

        # Note, does not encode position information (absolute or relative)
        self.temperature = temperature
        self.att_dim = attention_channels

        self.key_fun = nn.Linear(in_channels, attention_channels)
        self.value_fun = nn.Linear(in_channels, attention_channels)
        self.query_fun = nn.Linear(in_channels, attention_channels)
        self.pro_fun = nn.Linear(attention_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, expected shape `(B, C, H, W)`, where H is typically
            squeezed (e.g., H=1 after a `VerticalConv` or pooling) before
            applying attention along the W (time) dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input `(B, C, H, W)`, where
            attention has been applied across the W dimension.

        Raises
        ------
        RuntimeError
            If input tensor dimensions are incompatible with operations.
        """

        x = x.squeeze(2).permute(0, 2, 1)

        key = torch.matmul(
            x, self.key_fun.weight.T
        ) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        query = torch.matmul(
            x, self.query_fun.weight.T
        ) + self.query_fun.bias.unsqueeze(0).unsqueeze(0)
        value = torch.matmul(
            x, self.value_fun.weight.T
        ) + self.value_fun.bias.unsqueeze(0).unsqueeze(0)

        kk_qq = torch.bmm(key, query.permute(0, 2, 1)) / (
            self.temperature * self.att_dim
        )
        att_weights = F.softmax(kk_qq, 1)
        att = torch.bmm(value.permute(0, 2, 1), att_weights)

        op = torch.matmul(
            att.permute(0, 2, 1), self.pro_fun.weight.T
        ) + self.pro_fun.bias.unsqueeze(0).unsqueeze(0)
        op = op.permute(0, 2, 1).unsqueeze(2)

        return op

    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2).permute(0, 2, 1)

        key = torch.matmul(
            x, self.key_fun.weight.T
        ) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        query = torch.matmul(
            x, self.query_fun.weight.T
        ) + self.query_fun.bias.unsqueeze(0).unsqueeze(0)

        kk_qq = torch.bmm(key, query.permute(0, 2, 1)) / (
            self.temperature * self.att_dim
        )
        att_weights = F.softmax(kk_qq, 1)
        return att_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for temporal modeling.

    This module implements a multi-head scaled dot-product self-attention,
    extending the single-head `SelfAttention` to capture different aspects
    of temporal relationships in parallel. Multiple attention heads can learn
    to attend to different patterns or time scales simultaneously.

    Each head independently computes attention over the time dimension, and
    their outputs are concatenated and projected to the output space. This
    is particularly useful in bottleneck layers where rich temporal context
    is crucial for downstream reconstruction tasks.

    Parameters
    ----------
    in_channels : int
        Number of input channels (features per time step).
    attention_channels : int
        Total dimensionality of the multi-head attention space. Should be
        divisible by `num_heads`. Each head will have dimension
        `attention_channels // num_heads`.
    num_heads : int, default=4
        Number of parallel attention heads.
    dropout : float, default=0.0
        Dropout probability applied to attention weights.
    temperature : float, default=1.0
        Scaling factor for attention scores.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimensionality per attention head.
    attention_channels : int
        Total attention dimensionality.
    scale : float
        Scaling factor for attention scores (1 / sqrt(head_dim)).
    qkv_proj : nn.Linear
        Combined projection layer for query, key, and value.
    out_proj : nn.Linear
        Output projection layer.
    dropout : nn.Dropout
        Dropout layer for attention weights.

    Raises
    ------
    ValueError
        If `attention_channels` is not divisible by `num_heads`.
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        if attention_channels % num_heads != 0:
            raise ValueError(
                f"attention_channels ({attention_channels}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = attention_channels // num_heads
        self.attention_channels = attention_channels
        self.temperature = temperature
        self.scale = (1.0 / (self.head_dim ** 0.5)) / temperature

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(in_channels, attention_channels * 3)
        self.out_proj = nn.Linear(attention_channels, in_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C, H, W)`, where H is typically 1
            (squeezed) and W is the time dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input `(B, C, H, W)`.
        """
        batch_size = x.shape[0]
        
        # Reshape: (B, C, H, W) -> (B, W, C)
        x_reshaped = x.squeeze(2).permute(0, 2, 1)
        seq_len = x_reshaped.shape[1]

        # Project to Q, K, V: (B, W, C) -> (B, W, 3 * attention_channels)
        qkv = self.qkv_proj(x_reshaped)
        
        # Reshape to separate heads: (B, W, 3*attn_ch) -> (B, W, 3, num_heads, head_dim)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        
        # Permute to: (3, B, num_heads, W, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Compute attention scores: (B, num_heads, W, W)
        # query: (B, num_heads, W, head_dim)
        # key: (B, num_heads, head_dim, W)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax: (B, num_heads, W, W)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (B, num_heads, W, head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads: (B, W, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.attention_channels
        )

        # Final projection: (B, W, C)
        output = self.out_proj(attn_output)

        # Reshape back: (B, W, C) -> (B, C, 1, W)
        output = output.permute(0, 2, 1).unsqueeze(2)

        return output

    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute and return attention weights for visualization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Attention weights, shape `(B, num_heads, W, W)`.
        """
        batch_size = x.shape[0]
        x_reshaped = x.squeeze(2).permute(0, 2, 1)
        seq_len = x_reshaped.shape[1]

        qkv = self.qkv_proj(x_reshaped)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        
        query, key = qkv[0], qkv[1]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        return attn_weights


class LiteMLAConfig(BaseConfig):
    name: Literal["LiteMLA"] = "LiteMLA"
    out_channels: int
    dim_qk: int = 8
    dim_v: int = 16
    expansion_ratio: float = 3.0
    beta: float = 1.0
    use_bias: bool = False


class LiteMLA(nn.Module):
    """Lightweight Multi-head Linear Attention (LiteMLA).

    An efficient linear attention mechanism using ReLU activation instead of
    softmax, making it faster and more memory-efficient than standard attention.
    This is particularly useful for processing long sequences or when running
    on resource-constrained devices.

    The mechanism uses:
    - ReLU-activated query and key vectors (instead of softmax)
    - Linear complexity O(N) instead of quadratic O(N²)
    - Normalization trick for stable "probability-like" outputs

    Commonly used in efficient vision transformers like EfficientViT and
    MobileViT architectures. Ideal for encoder bottlenecks before transmission
    in split-device scenarios.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dim_qk : int, default=8
        Dimensionality for query and key projections per head.
    dim_v : int, default=16
        Dimensionality for value projections per head.
    expansion_ratio : float, default=3.0
        Controls the total dimension relative to input channels.
    beta : float, default=1.0
        Scaling factor for computing number of heads.
    use_bias : bool, default=False
        Whether to use bias in convolutions.

    Attributes
    ----------
    num_heads : int
        Number of attention heads (computed from expansion_ratio and beta).
    dim_qk : int
        Query/Key dimension per head.
    dim_v : int
        Value dimension per head.
    qkv : nn.Module
        1x1 convolution for Q, K, V projection.
    proj : nn.Module
        1x1 convolution for output projection with batch norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_qk: int = 8,
        dim_v: int = 16,
        expansion_ratio: float = 3.0,
        beta: float = 1.0,
        use_bias: bool = False,
    ):
        super().__init__()

        # Compute number of heads based on channel budget
        num_heads = int(
            in_channels * expansion_ratio * beta // ((dim_qk * 2 + dim_v))
        )
        total_dim = num_heads * (dim_qk * 2 + dim_v)
        
        self.num_heads = num_heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.total_dim = total_dim

        # QKV projection
        self.qkv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                total_dim,
                kernel_size=1,
                bias=use_bias,
            ),
        )

        # Output projection with batch norm
        self.proj = nn.Sequential(
            nn.Conv2d(
                num_heads * dim_v,
                out_channels,
                kernel_size=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight linear attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, out_channels, H, W)`.
        """
        B, _, H, W = x.shape

        # Project to QKV
        qkv = self.qkv(x)

        # Reshape for multi-head attention
        # (B, total_dim, H, W) -> (B, num_heads, dim_per_head, H*W)
        qkv = qkv.reshape(
            B, self.num_heads, self.total_dim // self.num_heads, H * W
        )
        qkv = qkv.transpose(-1, -2)  # (B, num_heads, H*W, dim_per_head)

        # Split into Q, K, V
        q = qkv[..., 0 : self.dim_qk]
        k = qkv[..., self.dim_qk : 2 * self.dim_qk]
        v = qkv[..., 2 * self.dim_qk :]

        # Apply ReLU activation (linear attention)
        q = F.relu(q)
        k = F.relu(k)

        # Efficient linear attention: O(N) complexity
        # Standard attention: softmax(QK^T)V = O(N²)
        # Linear attention: Q(K^TV) = O(N)
        k_t = k.transpose(-1, -2)  # (B, num_heads, dim_qk, H*W)
        
        # Normalization trick: pad v with 1s
        v_padded = F.pad(v, (0, 1), mode="constant", value=1.0)
        
        # K^T @ V: (B, num_heads, dim_qk, dim_v+1)
        kv = torch.matmul(k_t, v_padded)
        
        # Q @ (K^T @ V): (B, num_heads, H*W, dim_v+1)
        out = torch.matmul(q, kv)
        
        # Normalize to get "probability-like" output in (0,1)
        # Use larger epsilon and simple addition for quantization stability
        # Avoid torch.clamp_min as it introduces Cast/Where ops that esp-ppq can't handle
        denominator = out[..., -1:] + 1e-4  # Use single epsilon value
        out = out[..., :-1] / denominator

        # Reshape back
        out = out.transpose(-1, -2)  # (B, num_heads, dim_v, H*W)
        out = out.reshape(B, self.num_heads * self.dim_v, H, W)

        # Project to output
        out = self.proj(out)
        
        return out


class VectorQuantizerConfig(BaseConfig):
    """Configuration for Vector Quantizer."""
    
    name: Literal["VectorQuantizer"] = "VectorQuantizer"
    """Discriminator field indicating the block type."""
    
    codebook_size: int = 512
    """Number of vectors in the codebook."""
    
    commitment_cost: float = 0.25
    """Weight for the commitment loss (encoder commitment to codebook)."""
    
    ema_decay: float = 0.99
    """Decay rate for EMA updates of the codebook."""
    
    epsilon: float = 1e-5
    """Small constant for numerical stability."""


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA-updated codebook.
    
    Implements vector quantization similar to VQ-VAE, where continuous latent
    vectors are quantized to the nearest vector in a learned codebook. Uses
    Exponential Moving Average (EMA) for stable codebook updates instead of
    gradient-based optimization.
    
    This is ideal for split-device architectures where the quantized indices
    can be transmitted with minimal bandwidth (~10 bits per spatial position)
    instead of full float32 vectors.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (embedding dimension).
    codebook_size : int, default=512
        Number of vectors in the codebook.
    commitment_cost : float, default=0.25
        Weight for the commitment loss term.
    ema_decay : float, default=0.99
        Decay rate for exponential moving average updates.
    epsilon : float, default=1e-5
        Small constant for numerical stability in EMA.
    
    Attributes
    ----------
    codebook : nn.Embedding
        The learnable codebook of shape (codebook_size, in_channels).
    ema_cluster_size : nn.Parameter
        Running average of cluster sizes for EMA update.
    ema_w : nn.Parameter
        Running average of cluster sums for EMA update.
    """
    
    def __init__(
        self,
        in_channels: int,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.embedding_dim = in_channels
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # Initialize codebook
        self.codebook = nn.Embedding(codebook_size, in_channels)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', self.codebook.weight.data.clone())
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Apply vector quantization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C, H, W)`.
        
        Returns
        -------
        quantized : torch.Tensor
            Quantized tensor, shape `(B, C, H, W)`.
        info : dict
            Dictionary containing:
            - 'loss': VQ loss (commitment + codebook loss for EMA)
            - 'perplexity': Measure of codebook usage
            - 'encodings': One-hot encodings, shape `(B*H*W, codebook_size)`
            - 'encoding_indices': Indices into codebook, shape `(B*H*W,)`
        """
        # Reshape input: (B, C, H, W) -> (B, H, W, C) -> (BHW, C)
        input_shape = x.shape
        original_x = x  # Keep original for straight-through estimator
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_flat.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        # (BHW, C) @ (C, K) -> (BHW, K)
        distances = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(x_flat, self.codebook.weight.t())
        )
        
        # Find nearest codebook vector
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.codebook.weight)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        
        # EMA codebook update (only during training)
        if self.training:
            # Update cluster size with EMA
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                    (1 - self.ema_decay) * encodings_sum
            
            # Laplace smoothing of cluster size
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n
            )
            
            # Update embeddings with EMA
            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw
            
            # Normalize embeddings
            self.codebook.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        
        # Calculate perplexity for codebook usage monitoring
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # VQ losses for detection-focused training
        # 1. Commitment loss: encoder commits to nearest codebook entry
        commitment_loss = F.mse_loss(quantized.detach(), x_flat.view_as(quantized))
        
        # 2. Codebook loss: helps even with EMA, ensures codebook stays close to embeddings
        codebook_loss = F.mse_loss(quantized, x_flat.view_as(quantized).detach())
        
        # 3. Perplexity regularization: encourage diverse codebook usage
        # Negative because we want to maximize perplexity
        perplexity_loss = -perplexity / self.codebook_size
        
        # Combined VQ loss with strong perplexity penalty to prevent collapse
        loss = (
            self.commitment_cost * commitment_loss +
            0.1 * codebook_loss +
            1.0 * perplexity_loss  # Strong penalty - forces diverse codebook usage
        )
        
        # Straight-through estimator: permute back to (B, C, H, W) and add residual
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        quantized = original_x + (quantized - original_x).detach()
        
        return quantized, {
            'loss': loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity_loss': perplexity_loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
        }


class VariationalVectorQuantizerConfig(BaseConfig):
    """Configuration for Variational Vector Quantizer."""
    
    name: Literal["VariationalVectorQuantizer"] = "VariationalVectorQuantizer"
    """Discriminator field indicating the block type."""
    
    codebook_size: int = 512
    """Number of vectors in the codebook."""
    
    commitment_cost: float = 0.25
    """Weight for the commitment loss."""
    
    kl_weight: float = 0.0001
    """Weight for the KL divergence loss."""
    
    ema_decay: float = 0.99
    """Decay rate for EMA updates of the codebook."""
    
    epsilon: float = 1e-5
    """Small constant for numerical stability."""


class VariationalVectorQuantizer(nn.Module):
    """Variational Vector Quantizer (VQ-VAE style).
    
    Combines vector quantization with a variational component, adding a
    KL divergence loss to encourage the latent distribution to match a
    prior. This provides a hybrid between VQ-VAE and traditional VAE.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    codebook_size : int, default=512
        Number of vectors in the codebook.
    commitment_cost : float, default=0.25
        Weight for the commitment loss.
    kl_weight : float, default=0.0001
        Weight for the KL divergence loss.
    ema_decay : float, default=0.99
        Decay rate for EMA updates.
    epsilon : float, default=1e-5
        Small constant for numerical stability.
    """
    
    def __init__(
        self,
        in_channels: int,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        kl_weight: float = 0.0001,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        
        # Mean and logvar projections for variational component
        self.fc_mu = nn.Conv2d(in_channels, in_channels, 1)
        self.fc_logvar = nn.Conv2d(in_channels, in_channels, 1)
        
        # VQ component
        self.vq = VectorQuantizer(
            in_channels=in_channels,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            epsilon=epsilon,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Apply variational vector quantization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C, H, W)`.
        
        Returns
        -------
        quantized : torch.Tensor
            Quantized tensor, shape `(B, C, H, W)`.
        info : dict
            Dictionary containing VQ info plus 'kl_loss'.
        """
        # Variational component
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Vector quantization
        quantized, vq_info = self.vq(z)
        
        # Combine losses
        total_loss = vq_info['loss'] + self.kl_weight * kl_loss
        vq_info['loss'] = total_loss
        vq_info['kl_loss'] = kl_loss
        
        return quantized, vq_info


class PhiNetConvBlockConfig(BaseConfig):
    name: Literal["PhiNetConvBlock"] = "PhiNetConvBlock"
    out_channels: int
    expanded_channels: int
    stride: int = 1
    kernel_size: int = 3
    has_se: bool = True
    dropout_rate: float = 0.05


class PhiNetConvBlock(nn.Module):
    """PhiNet Convolutional Block for efficient mobile architectures.

    Implements the PhiNet block design optimized for resource-constrained devices.
    Uses depthwise separable convolutions with optional Squeeze-and-Excitation,
    residual connections, and efficient activation functions.

    Architecture:
    1. Expansion: 1×1 conv to increase channels (if needed)
    2. Depthwise: k×k depthwise conv with stride
    3. SE Block: Channel attention (optional)
    4. Projection: 1×1 conv to output channels (linear)
    5. Residual: Add input if stride=1 and channels match

    This is ideal for encoder stages on low-power devices in split architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    expanded_channels : int
        Number of channels after expansion phase.
    out_channels : int
        Number of output channels.
    stride : int, default=1
        Stride for depthwise convolution. Use 2 for downsampling.
    kernel_size : int, default=3
        Kernel size for depthwise convolution.
    has_se : bool, default=True
        Whether to include Squeeze-and-Excitation block.
    dropout_rate : float, default=0.05
        Dropout probability applied before depthwise conv.

    Attributes
    ----------
    use_res : bool
        Whether to use residual connection.
    expand : nn.Module or None
        Expansion layer (1×1 conv + BN + activation).
    depthwise : nn.Module
        Depthwise convolution layer.
    se : nn.Module
        Squeeze-and-Excitation block or Identity.
    project : nn.Module
        Projection layer (1×1 conv + BN).
    """

    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        has_se: bool = True,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        self.use_res = stride == 1 and in_channels == out_channels

        # 1. Expansion phase
        self.expand = None
        if expanded_channels != in_channels:
            padding = 0
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels, eps=1e-3, momentum=0.999),
                nn.Hardswish(),
            )

        # 2. Depthwise phase
        self.dropout = nn.Dropout2d(dropout_rate)
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels, eps=1e-3, momentum=0.999),
            nn.Hardswish(),
        )

        # 3. Squeeze and Excitation
        if has_se:
            reduced = max(1, expanded_channels // 16)  # Much smaller SE
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, reduced, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(reduced, expanded_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        # 4. Projection phase (Linear)
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PhiNet block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H//stride, W//stride)`.
        """
        residual = x

        # Expansion
        if self.expand is not None:
            x = self.expand(x)

        # Depthwise with dropout
        x = self.dropout(x)
        x = self.depthwise(x)

        # Squeeze-and-Excitation
        if self.se is not None:
            x = x * self.se(x)

        # Projection
        x = self.project(x)

        # Residual connection
        if self.use_res:
            x = x + residual

        return x


class ConvConfig(BaseConfig):
    """Configuration for a basic ConvBlock."""

    name: Literal["ConvBlock"] = "ConvBlock"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class ConvBlock(nn.Module):
    """Basic Convolutional Block.

    A standard building block consisting of a 2D convolution, followed by
    batch normalization and a ReLU activation function.

    Sequence: Conv2d -> BatchNorm2d -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Amount of padding added to preserve spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H, W)`.
        """
        return F.relu_(self.batch_norm(self.conv(x)))


class DepthwiseSeparableConvConfig(BaseConfig):
    """Configuration for depthwise-separable convolution block."""
    
    name: Literal["DepthwiseSeparableConv"] = "DepthwiseSeparableConv"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise-separable convolution: Depthwise conv -> Pointwise conv -> BN -> ReLU.
    
    More efficient than standard convolution, using ~1/9th the operations for 3x3 kernels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()
        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            groups=in_channels,
        )
        # Pointwise: 1x1 conv to change channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu_(self.batch_norm(x))


class DepthwiseSeparableConvDownConfig(BaseConfig):
    """Configuration for depthwise-separable convolution with downsampling."""
    
    name: Literal["DepthwiseSeparableConvDown"] = "DepthwiseSeparableConvDown"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1


class DepthwiseSeparableConvDownBlock(nn.Module):
    """Depthwise-separable convolution with 2x2 max pooling downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        down_scale: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.down_scale = down_scale
        
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = F.max_pool2d(x, kernel_size=self.down_scale)
        x = self.pointwise(x)
        return F.relu_(self.batch_norm(x))


class DepthwiseSeparableConvTransposeUpConfig(BaseConfig):
    """Configuration for depthwise-separable upsampling with ConvTranspose2d."""
    
    name: Literal["DepthwiseSeparableConvTransposeUp"] = "DepthwiseSeparableConvTransposeUp"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1


class DepthwiseSeparableConvTransposeUpBlock(nn.Module):
    """Depthwise-separable convolution with ConvTranspose2d upsampling (ESP32-optimized)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.up_scale = up_scale
        
        # Hardware-accelerated upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        
        # Depthwise + pointwise
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu_(self.batch_norm(x))


class VerticalConv(nn.Module):
    """Convolutional layer that aggregates features across the entire height.

    Applies a 2D convolution using a kernel with shape `(input_height, 1)`.
    This collapses the height dimension (H) to 1 while preserving the width (W),
    effectively summarizing features across the full vertical extent (e.g.,
    frequency axis) at each time step. Followed by BatchNorm and ReLU.

    Useful for summarizing frequency information before applying operations
    along the time axis (like SelfAttention).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    input_height : int
        The height (H dimension) of the input tensor. The convolutional kernel
        will be sized `(input_height, 1)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(input_height, 1),
            padding=0,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Vertical Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`, where H must match the
            `input_height` provided during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, 1, W)`.
        """
        return F.relu_(self.bn(self.conv(x)))


class FreqCoordConvDownConfig(BaseConfig):
    """Configuration for a FreqCoordConvDownBlock."""

    name: Literal["FreqCoordConvDown"] = "FreqCoordConvDown"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class FreqCoordConvDownBlock(nn.Module):
    """Downsampling Conv Block incorporating Frequency Coordinate features.

    This block implements a downsampling step (Conv2d + MaxPool2d) commonly
    used in CNN encoders. Before the convolution, it concatenates an extra
    channel representing the normalized vertical coordinate (frequency) to the
    input tensor.

    The purpose of adding coordinate features is to potentially help the
    convolutional filters become spatially aware, allowing them to learn
    patterns that might depend on the relative frequency position within the
    spectrogram.

    Sequence: Concat Coords -> Conv -> MaxPool -> BatchNorm -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of output channels after the convolution.
    input_height : int
        Height (H dimension, frequency bins) of the input tensor to this block.
        Used to generate the coordinate features.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()

        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height)[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CoordF -> Conv -> MaxPool -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`, where H must match
            `input_height`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H/2, W/2)` (due to MaxPool).
        """
        freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3])
        x = torch.cat((x, freq_info), 1)
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.batch_norm(x), inplace=True)
        return x


class FreqCoordConvDownBroadcastConfig(BaseConfig):
    """Configuration for FreqCoordConvDownBroadcastBlock (no Tile in ONNX)."""

    name: Literal["FreqCoordConvDownBroadcast"] = "FreqCoordConvDownBroadcast"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class FreqCoordConvDownBroadcastBlock(nn.Module):
    """Downsampling Conv Block with Frequency Coordinates using broadcast.

    Similar to FreqCoordConvDownBlock but uses expand() instead of repeat()
    to avoid Tile operations in ONNX export, making it compatible with more
    deployment targets like ESP32.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of output channels after the convolution.
    input_height : int
        Height (H dimension, frequency bins) of the input tensor to this block.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super().__init__()

        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height)[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CoordF -> Conv -> MaxPool -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H/2, W/2)`.
        """
        # Check if coords are disabled (for ESP32 export - full int8 mode)
        if self.coords is not None:
            # Check if coords are already pre-expanded (for ONNX export)
            if self.coords.shape[0] == x.shape[0] and self.coords.shape[3] == x.shape[3]:
                # Already expanded, use directly
                freq_info = self.coords
            else:
                # Normal case: repeat along batch and width dimensions  
                freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3])
            x = torch.cat((x, freq_info), 1)
        # If coords is None, skip concatenation (conv layer already adjusted for 1-channel input)
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.batch_norm(x), inplace=True)
        return x


class StandardConvDownConfig(BaseConfig):
    """Configuration for a StandardConvDownBlock."""

    name: Literal["StandardConvDown"] = "StandardConvDown"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class StandardConvDownBlock(nn.Module):
    """Standard Downsampling Convolutional Block.

    A basic downsampling block consisting of a 2D convolution, followed by
    2x2 max pooling, batch normalization, and ReLU activation.

    Sequence: Conv -> MaxPool -> BN -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of output channels after the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
    ):
        super(StandardConvDownBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            stride=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Apply Conv -> MaxPool -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H/2, W/2)`.
        """
        x = F.max_pool2d(self.conv(x), 2, 2)
        return F.relu(self.batch_norm(x), inplace=True)


class FreqCoordConvUpConfig(BaseConfig):
    """Configuration for a FreqCoordConvUpBlock."""

    name: Literal["FreqCoordConvUp"] = "FreqCoordConvUp"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class FreqCoordConvUpBlock(nn.Module):
    """Upsampling Conv Block incorporating Frequency Coordinate features.

    This block implements an upsampling step  followed by a convolution,
    commonly used in CNN decoders. Before the convolution, it concatenates an
    extra channel representing the normalized vertical coordinate (frequency)
    of the *upsampled* feature map.

    The goal is to provide spatial awareness (frequency position) to the
    filters during the decoding/upsampling process.

    Sequence: Interpolate  -> Concat Coords -> Conv -> BatchNorm -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (before upsampling).
    out_channels : int
        Number of output channels after the convolution.
    input_height : int
        Height (H dimension, frequency bins) of the tensor *before* upsampling.
        Used to calculate the height for coordinate feature generation after
        upsampling.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    up_mode : str, default="bilinear"
        Interpolation mode for upsampling (e.g., "nearest", "bilinear",
        "bicubic").
    up_scale : Tuple[int, int], default=(2, 2)
        Scaling factor for height and width during upsampling
        (typically (2, 2)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super().__init__()

        self.up_scale = up_scale
        self.up_mode = up_mode
        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height * up_scale[0])[
                None, None, ..., None
            ],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Interpolate -> Concat Coords -> Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W_in)`, where H_in should match
            `input_height` used during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H_in * scale_h, W_in * scale_w)`.
        """
        op = F.interpolate(
            x,
            scale_factor=self.up_scale,
            mode=self.up_mode,
            align_corners=False,
        )
        freq_info = self.coords.repeat(op.shape[0], 1, 1, op.shape[3])
        op = torch.cat((op, freq_info), 1)
        op = self.conv(op)
        op = F.relu(self.batch_norm(op), inplace=True)
        return op

class FreqCoordConvUpBroadcastConfig(BaseConfig):
    """Configuration for FreqCoordConvUpBroadcastBlock (no Tile in ONNX)."""

    name: Literal["FreqCoordConvUpBroadcast"] = "FreqCoordConvUpBroadcast"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1
    up_mode: str = "bilinear"
    up_scale: Tuple[int, int] = (2, 2)


class FreqCoordConvUpBroadcastBlock(nn.Module):
    """Upsampling Conv Block with Frequency Coordinates using broadcast."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.coords = nn.Parameter(
            torch.linspace(-1, 1, input_height * up_scale[0])[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size=kernel_size, padding=pad_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = F.interpolate(x, scale_factor=self.up_scale, mode=self.up_mode, align_corners=False)
        # Check if coords are disabled (for ESP32 export - full int8 mode)
        if self.coords is not None:
            # Check if coords are already pre-expanded (for ONNX export)
            # coords after upsample should match op.shape[3]
            if self.coords.shape[0] == op.shape[0] and self.coords.shape[3] == op.shape[3]:
                # Already expanded, use directly
                freq_info = self.coords
            else:
                # Normal case: repeat along batch and width dimensions
                freq_info = self.coords.repeat(op.shape[0], 1, 1, op.shape[3])
            op = torch.cat((op, freq_info), 1)
        # If coords is None, skip concatenation (conv layer already adjusted for 1-channel input)
        op = self.conv(op)
        op = F.relu(self.batch_norm(op), inplace=True)
        return op



class StandardConvUpConfig(BaseConfig):
    """Configuration for a StandardConvUpBlock."""

    name: Literal["StandardConvUp"] = "StandardConvUp"
    """Discriminator field indicating the block type."""

    out_channels: int
    """Number of output channels."""

    kernel_size: int = 3
    """Size of the square convolutional kernel."""

    pad_size: int = 1
    """Padding size."""


class StandardConvUpBlock(nn.Module):
    """Standard Upsampling Convolutional Block.

    A basic upsampling block used in CNN decoders. It first upsamples the input
    feature map using interpolation, then applies a 2D convolution, batch
    normalization, and ReLU activation. Does not use coordinate features.

    Sequence: Interpolate -> Conv -> BN -> ReLU.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (before upsampling).
    out_channels : int
        Number of output channels after the convolution.
    kernel_size : int, default=3
        Size of the square convolutional kernel.
    pad_size : int, default=1
        Padding added before convolution.
    up_mode : str, default="bilinear"
        Interpolation mode for upsampling (e.g., "nearest", "bilinear").
    up_scale : Tuple[int, int], default=(2, 2)
        Scaling factor for height and width during upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        up_mode: str = "bilinear",
        up_scale: Tuple[int, int] = (2, 2),
    ):
        super(StandardConvUpBlock, self).__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Interpolate -> Conv -> BN -> ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape `(B, C_in, H_in, W_in)`.

        Returns
        -------
        torch.Tensor
            Output tensor, shape `(B, C_out, H_in * scale_h, W_in * scale_w)`.
        """
        op = F.interpolate(
            x,
            scale_factor=self.up_scale,
            mode=self.up_mode,
            align_corners=False,
        )
        op = self.conv(op)
        op = F.relu(self.batch_norm(op), inplace=True)
        return op


# ============================================================================
# ESP32-Optimized Block Configs
# ============================================================================


class FreqCoordConvTransposeUpBroadcastConfig(BaseConfig):
    """Config for ESP32-optimized upsampling with ConvTranspose2d."""
    
    name: Literal["FreqCoordConvTransposeUpBroadcast"] = "FreqCoordConvTransposeUpBroadcast"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1


class FreqCoordConvTransposeUpConfig(BaseConfig):
    """Config for standard ESP32-optimized upsampling."""
    
    name: Literal["FreqCoordConvTransposeUp"] = "FreqCoordConvTransposeUp"
    out_channels: int
    kernel_size: int = 3
    pad_size: int = 1


class LiteMLA_ESP32Config(BaseConfig):
    """Config for ESP32-optimized LiteMLA (uses multiplication instead of division)."""
    
    name: Literal["LiteMLA_ESP32"] = "LiteMLA_ESP32"
    out_channels: int
    dim_qk: int = 8
    dim_v: int = 16
    expansion_ratio: float = 2.0
    beta: float = 1.0


LayerConfig = Annotated[
    Union[
        ConvConfig,
        DepthwiseSeparableConvConfig,
        DepthwiseSeparableConvDownConfig,
        DepthwiseSeparableConvTransposeUpConfig,
        FreqCoordConvDownConfig,
        FreqCoordConvDownBroadcastConfig,
        StandardConvDownConfig,
        FreqCoordConvUpConfig,
        FreqCoordConvUpBroadcastConfig,
        StandardConvUpConfig,
        FreqCoordConvTransposeUpBroadcastConfig,
        FreqCoordConvTransposeUpConfig,
        SelfAttentionConfig,
        MultiHeadAttentionConfig,
        LiteMLAConfig,
        LiteMLA_ESP32Config,
        PhiNetConvBlockConfig,
        VectorQuantizerConfig,
        VariationalVectorQuantizerConfig,
        "XiConvConfig",
        "XiConvDownConfig",
        "XiConvUpConfig",
        "LayerGroupConfig",
    ],
    Field(discriminator="name"),
]
"""Type alias for the discriminated union of block configuration models."""


class LayerGroupConfig(BaseConfig):
    name: Literal["LayerGroup"] = "LayerGroup"
    layers: List[LayerConfig]


def build_layer_from_config(
    input_height: int,
    in_channels: int,
    config: LayerConfig,
) -> Tuple[nn.Module, int, int]:
    """Factory function to build a specific nn.Module block from its config.

    Takes configuration object (one of the types included in the `LayerConfig`
    union) and instantiates the corresponding nn.Module block with the correct
    parameters derived from the config and the current pipeline state
    (`input_height`, `in_channels`).

    It uses the `name` field within the `config` object to determine
    which block class to instantiate.

    Parameters
    ----------
    input_height : int
        Height (frequency bins) of the input tensor *to this layer*.
    in_channels : int
        Number of channels in the input tensor *to this layer*.
    config : LayerConfig
        A Pydantic configuration object for the desired block (e.g., an
        instance of `ConvConfig`, `FreqCoordConvDownConfig`, etc.), identified
        by its `name` field.

    Returns
    -------
    Tuple[nn.Module, int, int]
        A tuple containing:
        - The instantiated `nn.Module` block.
        - The number of output channels produced by the block.
        - The calculated height of the output produced by the block.

    Raises
    ------
    NotImplementedError
        If the `config.name` does not correspond to a known block type.
    ValueError
        If parameters derived from the config are invalid for the block.
    """
    if config.name == "ConvBlock":
        return (
            ConvBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height,
        )

    if config.name == "DepthwiseSeparableConv":
        return (
            DepthwiseSeparableConvBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height,
        )

    if config.name == "DepthwiseSeparableConvDown":
        return (
            DepthwiseSeparableConvDownBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height // 2,
        )

    if config.name == "DepthwiseSeparableConvTransposeUp":
        return (
            DepthwiseSeparableConvTransposeUpBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "FreqCoordConvDown":
        return (
            FreqCoordConvDownBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height // 2,
        )

    if config.name == "FreqCoordConvDownBroadcast":
        return (
            FreqCoordConvDownBroadcastBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height // 2,
        )

    if config.name == "StandardConvDown":
        return (
            StandardConvDownBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height // 2,
        )

    if config.name == "FreqCoordConvUp":
        return (
            FreqCoordConvUpBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "FreqCoordConvUpBroadcast":
        return (
            FreqCoordConvUpBroadcastBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "FreqCoordConvTransposeUpBroadcast":
        from batdetect2.models.blocks_esp32 import FreqCoordConvTransposeUpBroadcastBlock
        return (
            FreqCoordConvTransposeUpBroadcastBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "FreqCoordConvTransposeUp":
        from batdetect2.models.blocks_esp32 import FreqCoordConvTransposeUpBlock
        return (
            FreqCoordConvTransposeUpBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                input_height=input_height,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "StandardConvUp":
        return (
            StandardConvUpBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                pad_size=config.pad_size,
            ),
            config.out_channels,
            input_height * 2,
        )

    if config.name == "SelfAttention":
        return (
            SelfAttention(
                in_channels=in_channels,
                attention_channels=config.attention_channels,
                temperature=config.temperature,
            ),
            config.attention_channels,
            input_height,
        )

    if config.name == "MultiHeadAttention":
        return (
            MultiHeadAttention(
                in_channels=in_channels,
                attention_channels=config.attention_channels,
                num_heads=config.num_heads,
                dropout=config.dropout,
                temperature=config.temperature,
            ),
            in_channels,
            input_height,
        )

    if config.name == "LiteMLA":
        return (
            LiteMLA(
                in_channels=in_channels,
                out_channels=config.out_channels,
                dim_qk=config.dim_qk,
                dim_v=config.dim_v,
                expansion_ratio=config.expansion_ratio,
                beta=config.beta,
                use_bias=config.use_bias,
            ),
            config.out_channels,
            input_height,
        )

    if config.name == "LiteMLA_ESP32":
        from batdetect2.models.blocks_esp32 import LiteMLA_ESP32
        return (
            LiteMLA_ESP32(
                in_channels=in_channels,
                out_channels=config.out_channels,
                dim_qk=config.dim_qk,
                dim_v=config.dim_v,
                expansion_ratio=config.expansion_ratio,
                beta=config.beta,
            ),
            config.out_channels,
            input_height,
        )

    if config.name == "PhiNetConvBlock":
        # Calculate output height based on stride
        output_height = input_height // config.stride
        return (
            PhiNetConvBlock(
                in_channels=in_channels,
                expanded_channels=config.expanded_channels,
                out_channels=config.out_channels,
                stride=config.stride,
                kernel_size=config.kernel_size,
                has_se=config.has_se,
                dropout_rate=config.dropout_rate,
            ),
            config.out_channels,
            output_height,
        )

    if config.name == "XiConv":
        # Calculate output height based on stride and pool
        output_height = input_height // config.stride
        if config.pool:
            output_height = output_height // config.pool
        return (
            XiConv(
                c_in=in_channels,
                c_out=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                groups=config.groups,
                act=config.act,
                gamma=config.gamma,
                attention=config.attention,
                attention_k=config.attention_k,
                attention_lite=config.attention_lite,
                batchnorm=config.batchnorm,
                dropout_rate=config.dropout_rate,
                pool=config.pool,
            ),
            config.out_channels,
            output_height,
        )

    if config.name == "XiConvDown":
        # Downsampling always reduces height by stride factor
        output_height = input_height // config.stride
        return (
            XiConv(
                c_in=in_channels,
                c_out=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                gamma=config.gamma,
                attention=config.attention,
                attention_k=config.attention_k,
                attention_lite=config.attention_lite,
                batchnorm=config.batchnorm,
                dropout_rate=config.dropout_rate,
            ),
            config.out_channels,
            output_height,
        )

    if config.name == "XiConvUp":
        # Upsampling always doubles height
        output_height = input_height * 2
        return (
            XiConvUpBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                gamma=config.gamma,
                attention=config.attention,
                attention_k=config.attention_k,
                attention_lite=config.attention_lite,
                batchnorm=config.batchnorm,
                dropout_rate=config.dropout_rate,
            ),
            config.out_channels,
            output_height,
        )

    if config.name == "VectorQuantizer":
        return (
            VectorQuantizer(
                in_channels=in_channels,
                codebook_size=config.codebook_size,
                commitment_cost=config.commitment_cost,
                ema_decay=config.ema_decay,
                epsilon=config.epsilon,
            ),
            in_channels,  # VQ doesn't change channel count
            input_height,
        )

    if config.name == "VariationalVectorQuantizer":
        return (
            VariationalVectorQuantizer(
                in_channels=in_channels,
                codebook_size=config.codebook_size,
                commitment_cost=config.commitment_cost,
                kl_weight=config.kl_weight,
                ema_decay=config.ema_decay,
                epsilon=config.epsilon,
            ),
            in_channels,  # VQ doesn't change channel count
            input_height,
        )

    if config.name == "LayerGroup":
        current_channels = in_channels
        current_height = input_height

        blocks = []

        for block_config in config.layers:
            block, current_channels, current_height = build_layer_from_config(
                input_height=current_height,
                in_channels=current_channels,
                config=block_config,
            )
            blocks.append(block)

        return nn.Sequential(*blocks), current_channels, current_height

    raise NotImplementedError(f"Unknown block type {config.name}")


def autopad(k: int, p: Optional[int] = None):
    """Implements padding to mimic 'same' behaviour for XiConv.
    
    Arguments
    ---------
    k : int
        Kernel size for the convolution.
    p : Optional[int]
        Padding value to be applied.
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class XiConv(nn.Module):
    """XiNet's convolutional block with compression and attention.

    Arguments
    ---------
    c_in: int
        Number of input channels.
    c_out: int
        Number of output channels.
    kernel_size: Union[int, Tuple]
        Kernel size for the main convolution.
    stride: Union[int, Tuple]
        Stride for the main convolution.
    padding: Optional[Union[int, Tuple]]
        Padding that is applied in the main convolution.
    groups: Optional[int]
        Number of groups for the main convolution.
    act: Optional[bool]
        When True, uses SiLU activation function.
    gamma: Optional[float]
        Compression factor for the convolutional block.
    attention: Optional[bool]
        When True, uses attention.
    skip_tensor_in: Optional[bool]
        When True, defines broadcasting skip connection block.
    skip_res : Optional[List]
        Spatial resolution of the skip connection.
    skip_channels: Optional[int]
        Number of channels for the input block.
    pool: Optional[bool]
        When True, applies pooling after the main convolution.
    attention_k: Optional[int]
        Kernel for the attention module.
    attention_lite: Optional[bool]
        When True, uses efficient attention implementation.
    batchnorm: Optional[bool]
        When True, uses batch normalization.
    dropout_rate: Optional[int]
        Dropout probability.
    skip_k: Optional[int]
        Kernel for the broadcast skip connection.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = True,
        gamma: Optional[float] = 4,
        attention: Optional[bool] = True,
        skip_tensor_in: Optional[bool] = False,
        skip_res: Optional[List[int]] = None,
        skip_channels: Optional[int] = 1,
        pool: Optional[Union[int, Tuple[int, int]]] = None,
        attention_k: Optional[int] = 3,
        attention_lite: Optional[bool] = True,
        batchnorm: Optional[bool] = True,
        dropout_rate: Optional[float] = 0,
        skip_k: Optional[int] = 1,
    ):
        super().__init__()
        self.compression = int(gamma)
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c_out // self.compression // 2
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.skip_tensor_in = skip_tensor_in

        if skip_tensor_in:
            assert skip_res is not None, "Specify shape of skip tensor."
            self.adaptive_pooling = nn.AdaptiveAvgPool2d(
                (int(skip_res[0]), int(skip_res[1]))
            )

        if self.compression > 1:
            self.compression_conv = nn.Conv2d(
                c_in, c_out // self.compression, 1, 1, groups=groups, bias=False
            )
        self.main_conv = nn.Conv2d(
            c_out // self.compression if self.compression > 1 else c_in,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=autopad(kernel_size, padding),
            bias=False,
        )
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        if attention:
            if attention_lite:
                self.att_pw_conv = nn.Conv2d(
                    c_out, self.attention_lite_ch_in, 1, 1, groups=groups, bias=False
                )
            self.att_conv = nn.Conv2d(
                c_out if not attention_lite else self.attention_lite_ch_in,
                c_out,
                attention_k,
                1,
                groups=groups,
                padding=autopad(attention_k, None),
                bias=False,
            )
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(
                skip_channels,
                c_out // self.compression,
                skip_k,
                1,
                groups=groups,
                padding=autopad(skip_k, None),
                bias=False,
            )
        if batchnorm:
            self.bn = nn.BatchNorm2d(c_out)
        if dropout_rate > 0:
            self.do = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """Forward step of XiConv block."""
        s = None
        # skip connection
        if isinstance(x, list):
            s = self.adaptive_pooling(x[1])
            s = self.skip_conv(s)
            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)

        if s is not None:
            x = x + s

        if self.pool:
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in = self.att_pw_conv(x)
            else:
                att_in = x
            y = self.att_act(self.att_conv(att_in))
            x = x * y

        if self.dropout_rate > 0:
            x = self.do(x)

        return x


class XiConvConfig(BaseConfig):
    """Configuration for XiConv block."""

    name: Literal["XiConv"] = "XiConv"
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] = None
    groups: int = 1
    act: bool = True
    gamma: float = 4.0
    attention: bool = True
    attention_k: int = 3
    attention_lite: bool = True
    batchnorm: bool = True
    dropout_rate: float = 0.0
    pool: Optional[int] = None


class XiConvDownConfig(BaseConfig):
    """Configuration for XiConv downsampling block."""

    name: Literal["XiConvDown"] = "XiConvDown"
    out_channels: int
    kernel_size: int = 3
    stride: int = 2
    gamma: float = 4.0
    attention: bool = True
    attention_k: int = 3
    attention_lite: bool = True
    batchnorm: bool = True
    dropout_rate: float = 0.0


class XiConvUpBlock(nn.Module):
    """XiConv upsampling block for decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        up_scale: Tuple[int, int] = (2, 2),
        gamma: float = 4.0,
        attention: bool = True,
        attention_k: int = 3,
        attention_lite: bool = True,
        batchnorm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.up_scale = up_scale
        self.xiconv = XiConv(
            c_in=in_channels,
            c_out=out_channels,
            kernel_size=kernel_size,
            stride=1,
            gamma=gamma,
            attention=attention,
            attention_k=attention_k,
            attention_lite=attention_lite,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample then apply XiConv."""
        x = F.interpolate(
            x,
            scale_factor=self.up_scale,
            mode="bilinear",
            align_corners=False,
        )
        return self.xiconv(x)


class XiConvUpConfig(BaseConfig):
    """Configuration for XiConv upsampling block."""

    name: Literal["XiConvUp"] = "XiConvUp"
    out_channels: int
    kernel_size: int = 3
    gamma: float = 4.0
    attention: bool = True
    attention_k: int = 3
    attention_lite: bool = True
    batchnorm: bool = True
    dropout_rate: float = 0.0

