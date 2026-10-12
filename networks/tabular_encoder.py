"""Transformer encoder for the five clinical indicators.

Each clinical indicator (sex, age, height, weight, BMI) is treated as a token.
A learnable <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token and per-column embeddings are prepended/added, then a
single Transformer block models interactions between indicators. The design
follows the FT-Transformer style encoder used in
https://github.com/siyi-wind/TIP.
"""

import math

import torch
import torch.nn as nn


def drop_path(x, drop_prob=0.0, training=False):
	"""Per-sample stochastic depth (DropPath)."""
	if drop_prob == 0.0 or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)
	random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	random_tensor.floor_()
	return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
	def __init__(self, drop_prob=0.0):
		super().__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
	"""In-place truncated-normal initialization (no timm dependency)."""
	with torch.no_grad():
		nn.init.normal_(tensor, mean=mean, std=std)
		while True:
			violations = (tensor < a) | (tensor > b)
			if not violations.any():
				break
			tensor[violations] = torch.empty_like(tensor[violations]).normal_(mean, std)
	return tensor


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None,
				 act_layer=nn.GELU, drop=0.0):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
		super().__init__()
		assert dim % num_heads == 0, 'dim must be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		batch, n_tokens, dim = x.shape
		qkv = self.qkv(x).reshape(
			batch, n_tokens, 3, self.num_heads, self.head_dim
		).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(batch, n_tokens, dim)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class TransformerBlock(nn.Module):
	def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False,
				 drop=0.1, attn_drop=0.0, drop_path=0.0,
				 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = MultiHeadSelfAttention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias,
			attn_drop=attn_drop, proj_drop=drop
		)
		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
		self.norm2 = norm_layer(dim)
		self.mlp = Mlp(
			in_features=dim, hidden_features=int(dim * mlp_ratio),
			act_layer=act_layer, drop=drop
		)

	def forward(self, x):
		x = x + self.drop_path(self.attn(self.norm1(x)))
		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class TabularEncoder(nn.Module):
	"""Encode the clinical indicator vector into a sequence of column tokens.

	Input : (B, 1, num_features) normalized clinical indicators.
	Output: (B, num_features + 1, embed_dim) token sequence, where token 0 is
	the global <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token and tokens 1..num_features are per-indicator.
	"""

	def __init__(self, num_features=5, embed_dim=64, num_layers=1, num_heads=8,
				 drop_rate=0.1):
		super().__init__()
		self.num_features = num_features
		self.embed_dim = embed_dim

		# Every indicator is a scalar, projected to the embedding dimension.
		self.value_proj = nn.Linear(1, embed_dim)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.column_embedding = nn.Embedding(num_features + 1, embed_dim)
		self.register_buffer(
			'position_ids', torch.arange(num_features + 1).expand(1, -1), persistent=False
		)

		self.norm = nn.LayerNorm(embed_dim)
		self.blocks = nn.ModuleList([
			TransformerBlock(dim=embed_dim, num_heads=num_heads, drop=drop_rate)
			for _ in range(num_layers)
		])

		trunc_normal_(self.cls_token, std=0.02)
		self.apply(self._init_weights)

	@staticmethod
	def _init_weights(module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, x):
		# (B, 1, F) -> (B, F, 1): one token per clinical indicator.
		x = x.permute(0, 2, 1)
		x = self.value_proj(x)                                  # (B, F, E)

		cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat([cls_tokens, x], dim=1)                  # (B, F+1, E)
		x = x + self.column_embedding(self.position_ids)
		x = self.norm(x)

		for block in self.blocks:
			x = block(x)
		return x
