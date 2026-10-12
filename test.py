"""Evaluate saved fold checkpoints and aggregate cross-validation metrics.

Example:
    python test.py --data_csv data/sarcopenia_processed.csv \
        --split_file data/3_fold_splits.pkl --image_dir data/images_224 \
        --checkpoint_dir runs/himamba/checkpoints
"""

import os

import numpy as np
import torch

from dataloaders.dataset import build_dataloaders
from networks.himamba import load_model_from_checkpoint
from utils.checkpoint import get_checkpoint_path
from utils.engine import evaluate
from utils.metrics import icc_3_1, plot_predictions, regression_metrics
from utils.parser import get_args


def test_fold(args, fold, device, checkpoint_dir):
	_, _, test_loader, _ = build_dataloaders(args, fold)

	checkpoint_path = get_checkpoint_path(checkpoint_dir, fold)
	model = load_model_from_checkpoint(checkpoint_path, args).to(device)

	labels, predictions = evaluate(model, test_loader, device)
	figure_dir = os.path.join(args.save_dir, 'figures')
	plot_predictions(labels, predictions, fold, figure_dir)

	metrics = regression_metrics(labels, predictions)
	icc = icc_3_1(labels, predictions)
	if icc is not None:
		metrics['ICC'] = float(icc)
	return metrics


def main():
	args = get_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	checkpoint_dir = args.checkpoint_dir or os.path.join(args.save_dir, 'checkpoints')
	fold_metrics = []
	for fold in range(args.n_folds):
		metrics = test_fold(args, fold, device, checkpoint_dir)
		fold_metrics.append(metrics)
		print(f'Fold {fold}: {metrics}')

	print('\n================ Test summary ================')
	names = ['PCC', 'MAE', 'R2'] + (['ICC'] if 'ICC' in fold_metrics[0] else [])
	for name in names:
		values = np.array([m[name] for m in fold_metrics])
		print(f'{name}: {values.mean():.4f} +/- {values.std():.4f}')


if __name__ == '__main__':
	main()
