"""Sarcopenia dataset and dataloader construction.

Expected processed CSV (see dataloaders/preprocess.py) contains at least:
    index            : unique case id used for logging
    path             : original DICOM path (its parent folder names the case)
    AGE, Gender, Height, Weight, BMI : min-max normalized clinical indicators
    ASMI             : raw ASMI label
    ASMI_Normalized  : min-max normalized ASMI label (training target)
    Cls_out          : sarcopenia category (kept for cohort statistics)

Images are pre-resized to 224x224 and stored as ``<image_dir>/<case_id>.npy``
by dataloaders/preprocess.py.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tvt
from monai.transforms import (
	Compose,
	NormalizeIntensityd,
	RandFlipd,
	RandRotated,
	RandZoomd,
	ScaleIntensityd,
)
from torch.utils.data import DataLoader, Dataset

from dataloaders.auto_augment import AutoAugment

CLINICAL_COLUMNS = ['AGE', 'Gender', 'Height', 'Weight', 'BMI']
LABEL_COLUMN = 'ASMI_Normalized'
RAW_LABEL_COLUMN = 'ASMI'
IMAGE_SIZE = 224

# Photometric augmentation (operates on a PIL image).
train_photometric_transform = tvt.Compose([
	tvt.ToPILImage(),
	AutoAugment(),
	tvt.ToTensor(),
])

# Geometric augmentation shared by the image (MONAI, applied after photometry).
train_geometric_transform = Compose([
	RandRotated(keys=['image'], range_x=10, range_y=10, prob=0.5),
	RandZoomd(keys=['image'], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
	RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
	RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
])

# Intensity standardization to [0, 1].
normalize_transform = Compose([
	NormalizeIntensityd(keys=['image'], nonzero=True),
	ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
])


def case_id_from_path(path):
	"""Extract the case id (DICOM parent folder name) from a stored path."""
	normalized = str(path).replace('\\', '/').rstrip('/')
	parts = normalized.split('/')
	if os.path.splitext(parts[-1])[1]:              # last part is a file name
		return parts[-2]
	return parts[-1]


class SarcopeniaDataset(Dataset):
	"""Dataset yielding a preprocessed X-ray, its clinical vector and label."""

	def __init__(self, records, image_dir, training=True):
		self.records = records
		self.image_dir = image_dir
		self.training = training

	def __len__(self):
		return len(self.records)

	def __getitem__(self, index):
		row = self.records[index]
		case_id = case_id_from_path(row['path'])
		image = np.load(os.path.join(self.image_dir, f'{case_id}.npy'))

		sample = {'image': image}
		if self.training:
			# .npy images are stored as CHW; ToPILImage expects HWC uint8.
			image_hwc = np.transpose(image, (1, 2, 0))
			sample['image'] = train_photometric_transform(image_hwc)
			sample = train_geometric_transform(sample)
		sample = normalize_transform(sample)
		image_tensor = sample['image'].as_tensor() if hasattr(sample['image'], 'as_tensor') \
			else torch.as_tensor(sample['image'])

		clinical = np.asarray(row[CLINICAL_COLUMNS], dtype=np.float32)
		clinical_tensor = torch.from_numpy(clinical).reshape(1, -1)      # (1, 5)
		label = torch.tensor(float(row[LABEL_COLUMN]), dtype=torch.float32)

		return {
			'image': image_tensor.float(),
			'clinical': clinical_tensor,
			'label': label,
			'case_id': case_id,
		}


def collate_batch(batch):
	images = torch.stack([item['image'] for item in batch], dim=0)
	clinical = torch.stack([item['clinical'] for item in batch], dim=0)
	labels = torch.stack([item['label'] for item in batch], dim=0)
	case_ids = [item['case_id'] for item in batch]
	return {'image': images, 'clinical': clinical, 'label': labels, 'case_id': case_ids}


def _select_records(dataframe, indices):
	return [row for _, row in dataframe[dataframe['index'].isin(indices)].iterrows()]


def check_fold_disjoint(split):
	train_set, val_set, test_set = set(split['train_idx']), set(split['val_idx']), set(split['test_idx'])
	assert not train_set & val_set, 'train/val overlap'
	assert not train_set & test_set, 'train/test overlap'
	assert not val_set & test_set, 'val/test overlap'


def build_dataloaders(args, fold):
	"""Construct train/val/test DataLoaders for one cross-validation fold."""
	splits = pd.read_pickle(args.split_file)
	split = splits[int(fold)]
	check_fold_disjoint(split)

	dataframe = pd.read_csv(args.data_csv)
	train_records = _select_records(dataframe, split['train_idx'])
	val_records = _select_records(dataframe, split['val_idx'])
	test_records = _select_records(dataframe, split['test_idx'])
	print(f'Fold {fold}: train {len(train_records)}, val {len(val_records)}, test {len(test_records)}')

	train_dataset = SarcopeniaDataset(train_records, args.image_dir, training=True)
	val_dataset = SarcopeniaDataset(val_records, args.image_dir, training=False)
	test_dataset = SarcopeniaDataset(test_records, args.image_dir, training=False)

	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, collate_fn=collate_batch,
		drop_last=(len(train_records) % args.batch_size == 1),
	)
	val_loader = DataLoader(
		val_dataset, batch_size=args.val_batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_batch,
	)
	test_loader = DataLoader(
		test_dataset, batch_size=args.val_batch_size, shuffle=False,
		num_workers=args.num_workers, collate_fn=collate_batch,
	)
	return train_loader, val_loader, test_loader, test_records


def move_batch(device, batch):
	"""Move a collated batch to the target device."""
	images = batch['image'].to(device, dtype=torch.float32)
	clinical = batch['clinical'].to(device, dtype=torch.float32)
	labels = batch['label'].to(device, dtype=torch.float32)
	return images, clinical, labels, batch['case_id']
