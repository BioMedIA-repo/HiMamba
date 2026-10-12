"""Data loading and preprocessing package.

Submodules are imported explicitly (e.g. ``from dataloaders.dataset import
build_dataloaders``) so that lightweight CLI tools such as
``dataloaders.make_splits`` do not pull in the full training stack.
"""

__all__ = ['dataset', 'preprocess', 'make_splits', 'auto_augment']
