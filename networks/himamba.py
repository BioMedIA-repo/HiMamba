"""HiMamba: Hierarchical State Space framework for asymmetric multimodal fusion.

Given a hip X-ray image and five clinical indicators (sex, age, height,
weight, BMI), HiMamba fuses the two asymmetric modalities in two stages:

1. Fine-grained Aligned Mamba Fusion (FAMF): each local clinical token is
   matched to image spatial tokens by L2-normalized cosine similarity, the
   aligned pairs are interleaved and processed by a Mamba block;
2. Global Clinical Attentive Screening (GCAS): the global clinical token
   cross-attends to image tokens to screen task-relevant regions.

The two fused vectors are concatenated and mapped to the ASMI regression value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.image_encoder import ResNetEncoder
from networks.mamba_block import MambaBlock
from networks.tabular_encoder import TabularEncoder


class FineGrainedAlignment(nn.Module):
	"""Align image spatial tokens to local clinical tokens (Eq. 1-3 in paper).

	A soft alignment matrix is built from the cosine similarity between
	L2-normalized tokens and a temperature-controlled softmax over image
	locations; image tokens are then aggregated per clinical indicator.
	"""

	def __init__(self, temperature=0.5):
		super().__init__()
		self.temperature = temperature

	def forward(self, image_tokens, clinical_tokens):
		"""
		Args:
			image_tokens:    (B, N_i, D) flattened image feature map tokens.
			clinical_tokens: (B, L, D) per-indicator clinical tokens.
		Returns:
			aligned:    (B, L, D) image features aligned to each indicator.
			similarity: (B, N_i, L) raw cosine similarity matrix.
		"""
		image_norm = F.normalize(image_tokens, p=2, dim=-1)
		clinical_norm = F.normalize(clinical_tokens, p=2, dim=-1)

		similarity = torch.matmul(image_norm, clinical_norm.transpose(-2, -1))
		align_weights = F.softmax(similarity / self.temperature, dim=1)
		aligned = torch.matmul(align_weights.transpose(-2, -1), image_tokens)
		return aligned, similarity


class InterleavedMambaFusion(nn.Module):
	"""Interleave aligned image/clinical tokens and model them with Mamba."""

	def __init__(self, hidden_dim=512):
		super().__init__()
		self.mamba = MambaBlock(hidden_dim)

	def forward(self, aligned_image_tokens, clinical_tokens):
		"""
		Build the sequence [img_1, tab_1, img_2, tab_2, ...] and run Mamba.
		"""
		batch, length, dim = clinical_tokens.shape
		sequence = torch.stack(
			[aligned_image_tokens, clinical_tokens], dim=2
		)                                          # (B, L, 2, D)
		sequence = sequence.reshape(batch, length * 2, dim)
		fused = self.mamba(sequence)               # (B, 2L, D)
		return fused, sequence


class RegressionHead(nn.Module):
	"""Two-layer MLP regressor."""

	def __init__(self, in_dim=1024, hidden_dim=512, num_outputs=1):
		super().__init__()
		self.head = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, num_outputs),
		)

	def forward(self, x):
		return self.head(x)


class HiMamba(nn.Module):
	def __init__(self, backbone='resnet18', num_clinical_features=5,
				 tabular_embed_dim=64, hidden_dim=512, num_outputs=1,
				 alignment_temperature=0.5, pretrained_path=None, pretrained=False):
		super().__init__()
		self.hidden_dim = hidden_dim

		# Encoders.
		self.image_encoder = ResNetEncoder(
			backbone=backbone, pretrained_path=pretrained_path, pretrained=pretrained
		)
		self.tabular_encoder = TabularEncoder(
			num_features=num_clinical_features, embed_dim=tabular_embed_dim
		)
		self.tabular_proj = nn.Linear(tabular_embed_dim, hidden_dim)

		# Global Clinical Attentive Screening (GCAS).
		self.global_screening = nn.MultiheadAttention(
			embed_dim=hidden_dim, num_heads=1, batch_first=True
		)

		# Fine-grained Aligned Mamba Fusion (FAMF).
		self.fine_grained_alignment = FineGrainedAlignment(
			temperature=alignment_temperature
		)
		self.mamba_fusion = InterleavedMambaFusion(hidden_dim=hidden_dim)

		# Regression head on the concatenation of the two fusion streams.
		self.regression_head = RegressionHead(
			in_dim=hidden_dim * 2, hidden_dim=hidden_dim, num_outputs=num_outputs
		)

	def get_backbone_layers(self):
		return [self.image_encoder]

	def optim_parameters(self, lr):
		"""Two parameter groups: pretrained backbone vs. newly added modules."""
		backbone_ids = [
			id(p) for module in self.get_backbone_layers()
			for p in module.parameters()
		]
		backbone_params = filter(lambda p: id(p) in backbone_ids, self.parameters())
		new_params = filter(lambda p: id(p) not in backbone_ids, self.parameters())
		return [
			{'params': backbone_params, 'lr': lr},
			{'params': new_params, 'lr': lr},
		]

	def forward(self, images, clinical, return_attention=False):
		"""
		Args:
			images:   (B, 3, 224, 224) preprocessed hip X-rays.
			clinical: (B, 1, 5) normalized clinical indicators.
			return_attention: if True, also return FAMF/GCAS attention maps.
		Returns:
			prediction:                 (B, 1) predicted ASMI.
			aligned_image_tokens:       (B, 5, D) for the InfoNCE loss.
			local_clinical_tokens:      (B, 5, D) for the InfoNCE loss.
			attention (optional):       dict with similarity / attention maps.
		"""
		# Image tokens: (B, D, 7, 7) -> (B, 49, D).
		image_feature_map = self.image_encoder(images)
		image_tokens = image_feature_map.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)

		# Clinical tokens: (B, 6, 64) -> (B, 6, D); token 0 is the global token.
		clinical_tokens = self.tabular_proj(self.tabular_encoder(clinical))
		global_clinical = clinical_tokens[:, 0:1, :]
		local_clinical = clinical_tokens[:, 1:, :]

		# GCAS: global clinical token queries the image tokens.
		global_fused, gcas_attention = self.global_screening(
			query=global_clinical, key=image_tokens, value=image_tokens
		)
		global_fused = global_fused.squeeze(dim=1)                  # (B, D)

		# FAMF: fine-grained alignment + interleaved Mamba fusion.
		aligned_image, similarity = self.fine_grained_alignment(image_tokens, local_clinical)
		local_fused, _ = self.mamba_fusion(aligned_image, local_clinical)
		local_fused = local_fused.mean(dim=1)                       # (B, D)

		fused = torch.cat([global_fused, local_fused], dim=1)       # (B, 2D)
		prediction = self.regression_head(fused)                    # (B, 1)

		if return_attention:
			attention = {
				'gcas_attention': gcas_attention,                   # (B, 1, 49)
				'famf_similarity': similarity,                      # (B, 49, 5)
				'famf_alignment': F.softmax(
					similarity / self.fine_grained_alignment.temperature, dim=1
				),
			}
			return prediction, aligned_image, local_clinical, attention
		return prediction, aligned_image, local_clinical


def build_model(args):
	"""Instantiate HiMamba from a parsed argument namespace/dict."""
	args = vars(args) if not isinstance(args, dict) else args
	return HiMamba(
		backbone=args.get('backbone', 'resnet18'),
		num_outputs=args.get('num_outputs', 1),
		alignment_temperature=args.get('alignment_temperature', 0.5),
		pretrained_path=args.get('pretrained_path', None),
		pretrained=args.get('pretrained', False),
	)


def load_model_from_checkpoint(checkpoint, args):
	"""Build a model and load a (possibly partial) state dict for evaluation."""
	model = build_model(args)
	if isinstance(checkpoint, str):
		checkpoint = torch.load(checkpoint, map_location='cpu')
	model.load_state_dict(checkpoint, strict=False)
	return model
