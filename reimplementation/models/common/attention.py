"""
MultiheadFlashAttention: Flash attention implementation for efficient transformer attention.
Pure PyTorch implementation without mmcv dependencies.
"""
import warnings
import math

import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_

from einops import rearrange
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_unpadded_kvpacked_func')
except:
    from flash_attn.flash_attn_interface import  flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_varlen_kvpacked_func')
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk, seqused = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


class MultiheadFlashAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (agent:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (agent:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(MultiheadFlashAttention, self).__init__()
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True
        self.attn = FlashMHA(
            embed_dim=embed_dims, 
            num_heads=num_heads, 
            attention_dropout=attn_drop, 
            dtype=torch.float16, 
            device='cuda',
            **kwargs
        )

        self.proj_drop = nn.Dropout(proj_drop)

        # Build dropout layer from config
        if dropout_layer and isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get('drop_prob', 0.)
            self.dropout_layer = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        else:
            self.dropout_layer = nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        assert attn_mask is None, 'attn mask not supported now.'
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # The dataflow('key', 'query', 'value') of ``FlashAttention`` is (batch, num_query, embed_dims).
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=key_padding_mask)[0]

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Generate sinusoidal positional embeddings.

    Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/

    Args:
        pos_tensor (Tensor): Position tensor of shape (..., 2) containing (x, y) coordinates
        hidden_dim (int): Dimension of the positional embedding. Default: 256

    Returns:
        Tensor: Positional embeddings of shape (..., hidden_dim)
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


import torch
import torch.nn as nn
import warnings
from functools import partial


def scaled_dot_product_attention_torch(q, k, v, scale=None, dropout_p=0.0):
    """
    Pure PyTorch scaled dot-product attention.
    Mirrors torch.nn.functional.scaled_dot_product_attention,
    but implemented explicitly for full control.
    """
    B, num_heads, N, head_dim = q.shape

    # (B, heads, N, N)
    attn = torch.matmul(q, k.transpose(-2, -1))
    if scale is not None:
        attn = attn * scale

    attn = attn.softmax(dim=-1)

    if dropout_p > 0:
        attn = nn.functional.dropout(attn, p=dropout_p)

    # (B, heads, N, head_dim)
    out = torch.matmul(attn, v)
    return out


class LayerScale(nn.Module):
    """Simple LayerScale implementation."""
    def __init__(self, dim, layer_scale_init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MultiheadAttention(nn.Module):
    """
    Pure-PyTorch rewrite of MMCV MultiheadAttention
    without any MMCV dependency.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.,
                 init_cfg=None):

        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        scale = qk_scale or (self.head_dims ** -0.5)

        # choose attention implementation
        self.scaled_dot_product_attention = partial(
            scaled_dot_product_attention_torch,
            scale=scale
        )

        # qkv projection
        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)

        self.attn_drop = attn_drop

        # output projection
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # dropout before residual connection
        self.out_drop = nn.Dropout(dropout_layer.get("drop_prob", 0.)
                                   if isinstance(dropout_layer, dict)
                                   else 0.)

        # layer scale
        if use_layer_scale or (layer_scale_init_value > 0):
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(embed_dims, layer_scale_init_value)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape

        # qkv projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)

        # merge heads
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        # output projection + layer scale + dropout
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        # optional v-shortcut
        if self.v_shortcut:
            x = v.mean(1) + x  # v: (B, heads, N, head_dim)

        return x


__all__ = ['MultiheadFlashAttention', 'gen_sineembed_for_position', 'MultiheadAttention']


def test_multihead_flash_attention():
    """Test MultiheadFlashAttention implementation."""
    import torch

    print("Testing MultiheadFlashAttention...")

    # Check if flash_attn is available
    try:
        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
            print("✓ flash_attn library found (using flash_attn_unpadded_kvpacked_func)")
        except ImportError:
            from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
            print("✓ flash_attn library found (using flash_attn_varlen_kvpacked_func)")
    except ImportError:
        print("⚠ flash_attn library not found - skipping tests")
        print("  Install with: pip install flash-attn")
        return

    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    embed_dims = 256
    num_heads = 8
    batch_size = 2
    num_queries = 100
    num_keys = 100

    attn = MultiheadFlashAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.1,
        proj_drop=0.1,
        batch_first=True
    )

    # Move to CUDA and half precision (required for flash attention)
    if torch.cuda.is_available():
        attn = attn.cuda().half()

        query = torch.randn(batch_size, num_queries, embed_dims).cuda().half()
        key = torch.randn(batch_size, num_keys, embed_dims).cuda().half()
        value = torch.randn(batch_size, num_keys, embed_dims).cuda().half()

        output = attn(query, key, value)
        print(f"   Input shape: {query.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == query.shape

        # Test 2: With positional encoding
        print("\n2. Testing with positional encoding...")
        query_pos = torch.randn(batch_size, num_queries, embed_dims).cuda().half()
        key_pos = torch.randn(batch_size, num_keys, embed_dims).cuda().half()

        output_pos = attn(query, key, value, query_pos=query_pos, key_pos=key_pos)
        print(f"   Output with pos encoding shape: {output_pos.shape}")
        assert output_pos.shape == query.shape

        # Test 3: With key padding mask
        print("\n3. Testing with key padding mask...")
        key_padding_mask = torch.rand(batch_size, num_keys).cuda() > 0.5
        output_mask = attn(query, key, value, key_padding_mask=key_padding_mask)
        print(f"   Output with padding mask shape: {output_mask.shape}")
        print(f"   Padding mask shape: {key_padding_mask.shape}, masked: {key_padding_mask.sum().item()}")

        # Test 4: Self-attention (key=value=query)
        print("\n4. Testing self-attention...")
        output_self = attn(query, query, query)
        print(f"   Self-attention output shape: {output_self.shape}")

        # Test 5: gen_sineembed_for_position
        print("\n5. Testing gen_sineembed_for_position...")
        pos_tensor = torch.rand(batch_size, num_queries, 2).cuda()
        pos_embed = gen_sineembed_for_position(pos_tensor, hidden_dim=embed_dims)
        print(f"   Position tensor shape: {pos_tensor.shape}")
        print(f"   Position embedding shape: {pos_embed.shape}")
        assert pos_embed.shape == (batch_size, num_queries, embed_dims)

        print("\n✓ All tests passed!")
    else:
        print("⚠ CUDA not available - skipping tests")
        print("  Flash attention requires CUDA")


if __name__ == '__main__':
    # To run this test: python -m reimplementation.models.common.attention
    test_multihead_flash_attention()

