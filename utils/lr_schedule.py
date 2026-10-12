"""Learning-rate schedule: linear warm-up followed by polynomial decay."""


def lr_warmup(base_lr, current_step, warmup_steps):
	return base_lr * (float(current_step) / max(1, warmup_steps))


def lr_poly(base_lr, current_step, max_steps, power=0.9):
	return base_lr * ((1 - float(current_step) / max_steps) ** power)


def adjust_learning_rate(optimizer, base_lr, current_step, max_steps,
						 backbone_group_index=0, new_module_lr_mult=10.0):
	"""Set per-group LR.

	The pretrained backbone (group 0) uses ``base_lr``; newly added modules
	(group 1) use ``new_module_lr_mult`` x base_lr, following the original
	training recipe.
	"""
	warmup_steps = max(1, max_steps // 20)
	if current_step < warmup_steps:
		backbone_lr = lr_warmup(base_lr, current_step, warmup_steps)
	else:
		backbone_lr = lr_poly(base_lr, current_step, max_steps)

	optimizer.param_groups[backbone_group_index]['lr'] = backbone_lr
	if len(optimizer.param_groups) > 1:
		optimizer.param_groups[1]['lr'] = backbone_lr * new_module_lr_mult
	return backbone_lr
