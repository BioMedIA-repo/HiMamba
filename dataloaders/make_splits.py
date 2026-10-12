"""Generate stratified-by-randomness three-fold train/val/test splits.

Each fold reserves 20% of cases for testing and 10% of the total for
validation (12.5% of the remaining 80%), reproducing the paper's protocol.
Indices refer to the ``index`` column of the processed CSV and are stored as a
pickled list of dicts: [{'train_idx', 'val_idx', 'test_idx'}, ...].

Example:
    python -m dataloaders.make_splits \
        --data_csv data/sarcopenia_processed.csv --output data/3_fold_splits.pkl
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEEDS = [42, 123, 456]


def main(args):
	dataframe = pd.read_csv(args.data_csv)
	case_index = dataframe['index'].values
	labels = dataframe['ASMI_Normalized'].values

	splits = []
	for fold, seed in enumerate(SPLIT_SEEDS[:args.n_folds]):
		# 20% held out for testing.
		_, _, temp_idx, test_idx = train_test_split(
			labels, case_index, test_size=0.2, random_state=seed
		)
		# 12.5% of the remaining 80% -> 10% of the total for validation.
		# Splitting the index array twice reproduces the same shuffle as the
		# original feature/label-based call (only length and seed matter).
		_, _, train_idx, val_idx = train_test_split(
			temp_idx, temp_idx, test_size=0.125, random_state=seed + 1
		)
		splits.append({
			'fold': fold,
			'seed_step1': seed,
			'seed_step2': seed + 1,
			'train_idx': list(train_idx),
			'val_idx': list(val_idx),
			'test_idx': list(test_idx),
		})
		print(f'Fold {fold}: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}')

	pd.to_pickle(splits, args.output)
	print(f'Saved {args.n_folds}-fold splits to {args.output}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_csv', type=str, required=True)
	parser.add_argument('--output', type=str, required=True)
	parser.add_argument('--n_folds', type=int, default=3)
	main(parser.parse_args())
