"""Selective State Space Model (Mamba) block.

A pure-PyTorch implementation of the Mamba selective scan introduced in
Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(2023). Only the components used by HiMamba's fine-grained fusion module are
kept: a 1-D causal convolution, input-dependent (selective) scan parameters
and a discretized selective scan written with PyTorch primitives.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelectiveScan(nn.Module):
	"""Input-dependent selective scan (S6).

	Given the sequence features ``x`` and the per-step dynamic parameters
	``delta``, ``B`` and ``C``, it discretizes the continuous state space
	model with zero-order hold and runs the recurrence:

		h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
		y_t = h_t @ C_t + D * x_t
	"""

	def __init__(self, d_model, d_state=16, dt_rank=None, dt_min=1e-3, dt_max=0.1):
		super().__init__()
		self.d_model = d_model
		self.d_state = d_state
		self.dt_rank = dt_rank or math.ceil(d_model / 16)

		# Projects the low-rank delta vector back to the model width.
		self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

		# S4D real initialization of the state matrix; kept in log space.
		self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state)))
		# Skip connection parameter.
		self.D = nn.Parameter(torch.ones(d_model))

		# Initialize delta bias so that time steps start in a reasonable range.
		dt = torch.exp(
			torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
		).clamp(min=dt_min)
		inv_dt = dt + torch.log(-torch.expm1(-dt))
		with torch.no_grad():
			self.dt_proj.bias.copy_(inv_dt)

	def forward(self, x, delta, B, C):
		"""
		Args:
			x:     (B, L, D) sequence features.
			delta: (B, L, dt_rank) raw time-step embeddings.
			B:     (B, L, N) dynamic input matrix.
			C:     (B, L, N) dynamic output matrix.
		Returns:
			(B, L, D) scanned sequence.
		"""
		batch, length, dim = x.shape

		dt = F.softplus(self.dt_proj(delta))               # (B, L, D)
		A = -torch.exp(self.A_log.float())                 # (D, N)

		# Discretized matrices (zero-order hold).
		dt_a = torch.einsum('bld,dn->bldn', dt, A)
		dt_b = torch.einsum('bld,bln->bldn', dt, B)

		hidden = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
		outputs = []
		for t in range(length):
			x_t = x[:, t, :].unsqueeze(-1)                 # (B, D, 1)
			hidden = hidden * torch.exp(dt_a[:, t]) + dt_b[:, t] * x_t
			y_t = torch.einsum('bdn,bn->bd', hidden, C[:, t])
			outputs.append(y_t)

		output = torch.stack(outputs, dim=1)               # (B, L, D)
		output = output + x * self.D
		return output


class MambaBlock(nn.Module):
	"""Mamba block: input projection -> causal conv1d -> selective scan -> gate."""

	def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
		super().__init__()
		self.d_model = d_model
		self.d_state = d_state
		self.d_conv = d_conv
		self.expand = expand
		self.d_inner = int(expand * d_model)
		self.dt_rank = math.ceil(d_model / 16)

		self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
		self.conv1d = nn.Conv1d(
			in_channels=self.d_inner,
			out_channels=self.d_inner,
			kernel_size=d_conv,
			groups=self.d_inner,
			padding=d_conv - 1,
			bias=True,
		)
		self.activation = nn.SiLU()
		self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
		self.selective_scan = SelectiveScan(self.d_inner, d_state, dt_rank=self.dt_rank)
		self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

	def forward(self, x):
		"""
		Args:
			x: (B, L, D) input token sequence.
		Returns:
			(B, L, D) output token sequence.
		"""
		_, length, _ = x.shape

		x_and_residual = self.in_proj(x)                  # (B, L, 2*d_inner)
		x_branch, residual = x_and_residual.split(
			[self.d_inner, self.d_inner], dim=-1
		)

		x_branch = rearrange(x_branch, 'b l d -> b d l')
		x_branch = self.conv1d(x_branch)[:, :, :length]   # causal conv: crop right padding
		x_branch = rearrange(x_branch, 'b d l -> b l d')
		x_branch = self.activation(x_branch)

		x_dbl = self.x_proj(x_branch)                     # (B, L, dt_rank + 2*N)
		delta, B, C = torch.split(
			x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
		)

		y = self.selective_scan(x_branch, delta, B, C)
		y = y * self.activation(residual)                 # SiLU gate
		return self.out_proj(y)
