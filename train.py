"""Train HiMamba with three-fold cross-validation.

Example:
    python train.py --data_csv data/sarcopenia_processed.csv \
        --split_file data/3_fold_splits.pkl --image_dir data/images_224 \
        --pretrained_path checkpoints/resnet18-f37072fd.pth --gpu 0
"""

import os
import random

import numpy as np
import torch
from torch.optim import Adam

from dataloaders.dataset import build_dataloaders
from networks.himamba import build_model, load_model_from_checkpoint
from utils.checkpoint import save_checkpoint
from utils.engine import evaluate, train_one_epoch
from utils.metrics import plot_predictions, regression_metrics
from utils.parser import get_args


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def train_fold(args, fold, device, writer=None):
	train_loader, val_loader, test_loader, _ = build_dataloaders(args, fold)

	model = build_model(args).to(device)
	optimizer = Adam(
		model.optim_parameters(args.lr),
		lr=args.lr, weight_decay=args.weight_decay,
	)
	scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

	best_pcc = -float('inf')
	bad_epochs = 0
	best_model_path = None
	steps_per_epoch = len(train_loader)
	total_steps = steps_per_epoch * args.epochs

	for epoch in range(args.epochs):
		train_loss = train_one_epoch(
			model, train_loader, optimizer, device, epoch, args,
			global_step=epoch * steps_per_epoch, total_steps=total_steps,
			scaler=scaler,
		)
		val_labels, val_preds = evaluate(model, val_loader, device)
		val_metrics = regression_metrics(val_labels, val_preds)

		if writer is not None:
			writer.add_scalar(f'fold_{fold}/train_loss', train_loss, epoch)
			for name, value in val_metrics.items():
				writer.add_scalar(f'fold_{fold}/val_{name}', value, epoch)
		print(f'[Fold {fold}] Epoch {epoch}: loss={train_loss:.4f}, '
			  f"val PCC={val_metrics['PCC']:.4f}, MAE={val_metrics['MAE']:.4f}, "
			  f"R2={val_metrics['R2']:.4f}")

		if val_metrics['PCC'] >= best_pcc:
			best_pcc = val_metrics['PCC']
			best_model_path, _ = save_checkpoint(
				os.path.join(args.save_dir, 'checkpoints'), fold, model, optimizer
			)
			bad_epochs = 0
		else:
			bad_epochs += 1
			if bad_epochs >= args.patience:
				print(f'[Fold {fold}] early stopping at epoch {epoch}.')
				break

	# Evaluate the best checkpoint on the held-out test fold.
	checkpoint = torch.load(best_model_path, map_location=device)
	best_model = load_model_from_checkpoint(checkpoint, args).to(device)
	test_labels, test_preds = evaluate(best_model, test_loader, device)
	figure_dir = os.path.join(args.save_dir, 'figures')
	plot_predictions(test_labels, test_preds, fold, figure_dir)
	test_metrics = regression_metrics(test_labels, test_preds)
	print(f'[Fold {fold}] test {test_metrics}')
	return test_metrics


def summarize(fold_metrics):
	print('\n================ Cross-validation summary ================')
	for fold, metrics in enumerate(fold_metrics):
		print(f"Fold {fold}: PCC={metrics['PCC']:.4f}, "
			  f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
	for name in ['PCC', 'MAE', 'R2']:
		values = np.array([m[name] for m in fold_metrics])
		print(f'{name}: {values.mean():.4f} +/- {values.std():.4f}')


def main():
	args = get_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	set_seed(args.seed)
	torch.backends.cudnn.benchmark = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')

	os.makedirs(args.save_dir, exist_ok=True)
	writer = None
	try:
		from torch.utils.tensorboard import SummaryWriter
		writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
	except ImportError:
		print('TensorBoard unavailable; skipping event logs.')

	fold_metrics = []
	for fold in range(args.n_folds):
		print(f'\n---------------- Fold {fold} ----------------')
		fold_metrics.append(train_fold(args, fold, device, writer))

	summarize(fold_metrics)
	if writer is not None:
		writer.close()


if __name__ == '__main__':
	main()
