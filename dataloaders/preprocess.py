"""Build the processed CSV and 224x224 .npy images from raw DICOM data.

Expected raw CSV columns (names configurable via CLI):
    ID      : unique case identifier
    AGE, Gender, Height, Weight, BMI : raw clinical indicators
    ASMI    : ground-truth ASMI value
For every case the first ``*.dcm`` file under ``<dicom_root>/<ID>/`` is used.

Outputs:
    <output_csv>                      : processed CSV with normalized columns,
                                        an ``index`` column and a ``path`` column
    <output_image_dir>/<ID>.npy       : CHW uint8 array resized to 224x224

Example:
    python -m dataloaders.preprocess \
        --raw_csv data/raw/clinical.csv --dicom_root data/raw/images \
        --output_csv data/sarcopenia_processed.csv \
        --output_image_dir data/images_224
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.transform import resize

CLINICAL_COLUMNS = ['AGE', 'Gender', 'Height', 'Weight', 'BMI']
IMAGE_SIZE = 224


def read_dicom(scan_path):
	"""Read a DICOM file and return an HWC uint8/float array."""
	image = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))
	if image.ndim == 4:
		image = image.squeeze()
	if image.ndim == 3:
		# CHW -> HWC; replicate grayscale to three channels if needed.
		image = np.transpose(image, (1, 2, 0))
		if image.shape[-1] == 1:
			image = np.repeat(image, 3, axis=-1)
	return image


def min_max_normalize(image):
	image = image.astype(np.float32)
	image = (image - image.min()) / (image.max() - image.min() + 1e-8)
	return image


def preprocess_image(dicom_path, image_size=IMAGE_SIZE):
	"""DICOM -> normalized -> resized CHW uint8 array."""
	image = read_dicom(dicom_path)
	image = min_max_normalize(image)
	image = (image * 255).astype(np.uint8)

	image = np.transpose(image, (2, 0, 1))                    # HWC -> CHW
	image = resize(
		image, (image.shape[0], image_size, image_size),
		order=0, mode='edge', cval=0, clip=True,
		preserve_range=True, anti_aliasing=False,
	).astype(np.uint8)
	return image


def min_max_column(series):
	return (series - series.min()) / (series.max() - series.min() + 1e-8)


def main(args):
	df = pd.read_csv(args.raw_csv)
	required = [args.id_col, 'ASMI'] + CLINICAL_COLUMNS
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise KeyError(f'Raw CSV is missing columns: {missing}')

	# Normalize clinical indicators and the regression target.
	for column in CLINICAL_COLUMNS:
		df[column] = min_max_column(df[column])
	df['ASMI_Normalized'] = min_max_column(df['ASMI'])

	os.makedirs(args.output_image_dir, exist_ok=True)
	paths = []
	for _, row in df.iterrows():
		case_id = str(row[args.id_col])
		dicom_candidates = sorted(glob(os.path.join(args.dicom_root, case_id, '*.dcm')))
		if not dicom_candidates:
			raise FileNotFoundError(f'No DICOM found for case {case_id}')
		dicom_path = dicom_candidates[0]

		image = preprocess_image(dicom_path, args.image_size)
		np.save(os.path.join(args.output_image_dir, f'{case_id}.npy'), image)
		paths.append(dicom_path)

	df['path'] = paths
	df = df.rename(columns={args.id_col: 'Name'})
	df.reset_index(drop=False, inplace=True)          # explicit ``index`` column
	df.to_csv(args.output_csv, index=False)
	print(f'Processed {len(df)} cases -> {args.output_csv}, images -> {args.output_image_dir}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--raw_csv', type=str, required=True)
	parser.add_argument('--dicom_root', type=str, required=True)
	parser.add_argument('--output_csv', type=str, required=True)
	parser.add_argument('--output_image_dir', type=str, required=True)
	parser.add_argument('--id_col', type=str, default='ID')
	parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
	main(parser.parse_args())
