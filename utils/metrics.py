"""Regression metrics and prediction visualization."""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


class AverageMeter:
	"""Track the running average of a scalar or vector quantity."""

	def __init__(self):
		self.reset()

	def reset(self):
		self.value = 0.0
		self.average = 0.0
		self.sum = 0.0
		self.count = 0

	def update(self, value, n=1):
		self.value = float(value)
		self.sum += self.value * n
		self.count += n
		self.average = self.sum / self.count


def regression_metrics(y_true, y_pred):
	"""Compute PCC, MAE and R2 for 1-D prediction arrays."""
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	if len(y_true) < 2:
		return {'PCC': 0.0, 'MAE': 0.0, 'R2': 0.0}

	pcc, _ = pearsonr(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	return {'PCC': float(pcc), 'MAE': float(mae), 'R2': float(r2)}


def icc_3_1(y_true, y_pred):
	"""ICC(3,1) between ground truth and predictions (requires pingouin).

	Returns None if pingouin is not installed.
	"""
	try:
		import pandas as pd
		import pingouin as pg
	except ImportError:
		return None

	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	n = len(y_true)
	ratings = np.empty(n * 2, dtype=np.float64)
	ratings[0::2] = y_true
	ratings[1::2] = y_pred
	data = pd.DataFrame({
		'Target': np.repeat(np.arange(n), 2),
		'Rater': np.tile(['GroundTruth', 'Model'], n),
		'Rating': ratings,
	})
	icc_table = pg.intraclass_corr(
		data=data, targets='Target', raters='Rater', ratings='Rating'
	)
	return float(icc_table.loc[2, 'ICC'])


def plot_predictions(y_true, y_pred, fold, output_dir):
	"""Save a predicted-vs-true scatter plot with the y=x reference line."""
	os.makedirs(output_dir, exist_ok=True)
	metrics = regression_metrics(y_true, y_pred)

	plt.figure(figsize=(6, 6))
	plt.scatter(y_true, y_pred, alpha=0.5, s=18, color=(0.08, 0.40, 0.63))
	low = min(y_true.min(), y_pred.min())
	high = max(y_true.max(), y_pred.max())
	plt.plot([low, high], [low, high], 'r--', linewidth=1.5, label='y = x')

	plt.text(
		0.05, 0.95,
		f"PCC = {metrics['PCC']:.4f}\nMAE = {metrics['MAE']:.4f}\nR2  = {metrics['R2']:.4f}",
		transform=plt.gca().transAxes, va='top',
		bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
	)
	plt.xlabel('Ground-truth ASMI')
	plt.ylabel('Predicted ASMI')
	plt.title(f'Fold {fold}')
	plt.legend(loc='lower right')
	plt.tight_layout()
	save_path = os.path.join(output_dir, f'fold_{fold}_prediction_scatter.png')
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()
	return save_path
