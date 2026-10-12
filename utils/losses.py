"""Loss functions."""

import torch
import torch.nn.functional as F


def symmetric_info_nce(image_tokens, clinical_tokens, temperature=0.07):
	"""Symmetric InfoNCE between aligned image and clinical token pairs.

	Args:
		image_tokens:    (B, L, D) image tokens aligned to clinical tokens.
		clinical_tokens: (B, L, D) matching local clinical tokens.
		temperature:     softmax temperature.
	Returns:
		Scalar loss averaged over the image->clinical and clinical->image
		directions. Positives are the B*L matched (i, i) token pairs.
	"""
	batch, length, dim = image_tokens.shape

	image_flat = image_tokens.reshape(-1, dim)
	clinical_flat = clinical_tokens.reshape(-1, dim)

	image_norm = F.normalize(image_flat, dim=1)
	clinical_norm = F.normalize(clinical_flat, dim=1)

	logits = torch.mm(image_norm, clinical_norm.T) / temperature
	targets = torch.arange(batch * length, device=image_tokens.device)

	loss_i2t = F.cross_entropy(logits, targets)
	loss_t2i = F.cross_entropy(logits.T, targets)
	return (loss_i2t + loss_t2i) / 2
