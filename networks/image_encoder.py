"""Image encoder based on a pretrained ResNet backbone.

Extracts the last convolutional feature map (B, 512, 7, 7 for a 224x224 input
with ResNet-18/34), which is later flattened into 49 spatial image tokens.
"""

from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchvision.models as tv_models

_RESNET_FACTORY = {
	'resnet18': tv_models.resnet18,
	'resnet34': tv_models.resnet34,
}


class ResNetEncoder(nn.Module):
	"""ResNet feature extractor without the global pooling / FC head."""

	def __init__(self, backbone='resnet18', pretrained_path=None, pretrained=False):
		super().__init__()
		if backbone not in _RESNET_FACTORY:
			raise ValueError(f'Unsupported backbone: {backbone}. '
							 f'Choose from {list(_RESNET_FACTORY)}.')
		resnet = _RESNET_FACTORY[backbone](weights=None)

		if pretrained:
			# torchvision will download ImageNet weights automatically.
			resnet = _RESNET_FACTORY[backbone](weights='DEFAULT')
		elif pretrained_path is not None:
			# Load a local ImageNet checkpoint (e.g. resnet18-f37072fd.pth).
			import os
			if not os.path.isfile(pretrained_path):
				warnings.warn(
					f'Pretrained checkpoint not found: {pretrained_path}; '
					'initializing the backbone randomly.'
				)
			else:
				checkpoint = torch.load(pretrained_path, map_location='cpu')
				if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
					checkpoint = checkpoint['state_dict']
				resnet.load_state_dict(checkpoint, strict=False)

		# stem = conv1 + bn1 + relu (named_children order: conv1, bn1, relu, maxpool, ...)
		self.stem = nn.Sequential(OrderedDict(list(resnet.named_children())[0:3]))
		self.maxpool = resnet.maxpool
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		self.out_channels = 512

	def forward(self, x):
		x = self.stem(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x
