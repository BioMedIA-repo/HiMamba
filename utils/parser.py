"""Command-line arguments shared by train.py and test.py."""

import argparse


def get_args():
	parser = argparse.ArgumentParser(
		description='HiMamba: hierarchical Mamba fusion for ASMI regression.'
	)

	# ---------------- Data ----------------
	data_group = parser.add_argument_group('data')
	data_group.add_argument('--data_csv', type=str, default='data/sarcopenia_processed.csv',
							help='CSV of processed cases (clinical features + ASMI labels).')
	data_group.add_argument('--split_file', type=str, default='data/3_fold_splits.pkl',
							help='Pickle holding train/val/test indices for every fold.')
	data_group.add_argument('--image_dir', type=str, default='data/images_224',
							help='Directory of preprocessed 224x224 .npy images named <case_id>.npy.')
	data_group.add_argument('--n_folds', type=int, default=3, help='Number of CV folds.')
	data_group.add_argument('--num_workers', type=int, default=8, help='DataLoader workers.')

	# ---------------- Model ----------------
	model_group = parser.add_argument_group('model')
	model_group.add_argument('--backbone', type=str, default='resnet18',
							choices=['resnet18', 'resnet34'], help='Image backbone.')
	model_group.add_argument('--pretrained_path', type=str,
							default='checkpoints/resnet18-f37072fd.pth',
							help='Local ImageNet checkpoint for the backbone. Ignored if missing.')
	model_group.add_argument('--pretrained', action='store_true',
							help='Download torchvision ImageNet weights instead of a local file.')
	model_group.add_argument('--num_outputs', type=int, default=1, help='Regression outputs.')
	model_group.add_argument('--alignment_temperature', type=float, default=0.5,
							help='Temperature tau of the FAMF soft alignment.')

	# ---------------- Optimization ----------------
	opt_group = parser.add_argument_group('optimization')
	opt_group.add_argument('--lr', type=float, default=9e-5, help='Base learning rate.')
	opt_group.add_argument('--weight_decay', type=float, default=5e-4)
	opt_group.add_argument('--epochs', type=int, default=100)
	opt_group.add_argument('--batch_size', type=int, default=64)
	opt_group.add_argument('--val_batch_size', type=int, default=64)
	opt_group.add_argument('--patience', type=int, default=30,
						   help='Early-stop patience based on validation PCC.')
	opt_group.add_argument('--grad_clip', type=float, default=4.0)
	opt_group.add_argument('--update_every', type=int, default=4,
						   help='Gradient accumulation steps.')
	opt_group.add_argument('--infonce_weight', type=float, default=0.1,
						   help='Weight lambda of the InfoNCE loss.')
	opt_group.add_argument('--infonce_temperature', type=float, default=0.07)

	# ---------------- Experiment ----------------
	exp_group = parser.add_argument_group('experiment')
	exp_group.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES.')
	exp_group.add_argument('--seed', type=int, default=112)
	exp_group.add_argument('--save_dir', type=str, default='runs/himamba',
						   help='Directory for checkpoints, logs and figures.')
	exp_group.add_argument('--checkpoint_dir', type=str, default=None,
						   help='[test] Directory with fold_<k>_best.pt. Defaults to save_dir.')

	args = parser.parse_args()
	return args
