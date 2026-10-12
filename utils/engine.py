"""Training and evaluation loops shared by train.py and test.py."""

import numpy as np
import torch
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataloaders.dataset import move_batch
from utils.losses import symmetric_info_nce
from utils.lr_schedule import adjust_learning_rate
from utils.metrics import AverageMeter

mse_loss = MSELoss()


def compute_loss(prediction, label, aligned_image, local_clinical, args):
	"""L = MSE + lambda * InfoNCE (Eq. 11 in the paper)."""
	regression_loss = mse_loss(prediction, label)
	contrastive_loss = symmetric_info_nce(
		aligned_image, local_clinical, args.infonce_temperature
	)
	total_loss = regression_loss + args.infonce_weight * contrastive_loss
	return total_loss, regression_loss, contrastive_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, args,
					global_step, total_steps, scaler=None):
	model.train()
	loss_meter = AverageMeter()
	optimizer.zero_grad()
	use_amp = scaler is not None and device.type == 'cuda'

	progress = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)
	for step, batch in enumerate(progress):
		images, clinical, labels, _ = move_batch(device, batch)
		labels = labels.unsqueeze(1)

		adjust_learning_rate(
			optimizer, args.lr, epoch * len(dataloader) + step, total_steps
		)

		if use_amp:
			with torch.cuda.amp.autocast():
				prediction, aligned_image, local_clinical = model(images, clinical)
				loss, _, _ = compute_loss(
					prediction, labels, aligned_image, local_clinical, args
				)
			scaler.scale(loss).backward()
		else:
			prediction, aligned_image, local_clinical = model(images, clinical)
			loss, _, _ = compute_loss(
				prediction, labels, aligned_image, local_clinical, args
			)
			loss.backward()

		loss_meter.update(loss.item(), images.size(0))

		is_optimizer_step = (step + 1) % args.update_every == 0 or step == len(dataloader) - 1
		if is_optimizer_step:
			if use_amp:
				scaler.unscale_(optimizer)
				clip_grad_norm_(
					[p for p in model.parameters() if p.requires_grad], args.grad_clip
				)
				scaler.step(optimizer)
				scaler.update()
			else:
				clip_grad_norm_(
					[p for p in model.parameters() if p.requires_grad], args.grad_clip
				)
				optimizer.step()
			optimizer.zero_grad()

		progress.set_postfix(loss=f'{loss_meter.average:.4f}')

	return loss_meter.average


@torch.no_grad()
def evaluate(model, dataloader, device):
	"""Return ground-truth labels and predictions as 1-D numpy arrays."""
	model.eval()
	all_labels, all_predictions = [], []
	for batch in tqdm(dataloader, desc='Evaluating', dynamic_ncols=True):
		images, clinical, labels, _ = move_batch(device, batch)
		prediction, _, _ = model(images, clinical)
		all_predictions.append(prediction.cpu().numpy().reshape(-1))
		all_labels.append(labels.cpu().numpy().reshape(-1))
	return np.concatenate(all_labels), np.concatenate(all_predictions)
