"""Checkpoint helpers."""

import os

import torch


def save_checkpoint(save_dir, fold, model, optimizer=None):
	"""Save the model (and optionally optimizer) for one CV fold."""
	os.makedirs(save_dir, exist_ok=True)
	model_path = os.path.join(save_dir, f'fold_{fold}_best.pt')
	torch.save(model.state_dict(), model_path)

	optimizer_path = None
	if optimizer is not None:
		optimizer_path = os.path.join(save_dir, f'fold_{fold}_best_optimizer.pt')
		torch.save(optimizer.state_dict(), optimizer_path)
	return model_path, optimizer_path


def get_checkpoint_path(checkpoint_dir, fold):
	return os.path.join(checkpoint_dir, f'fold_{fold}_best.pt')
