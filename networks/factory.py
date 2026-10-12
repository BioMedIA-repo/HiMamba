"""Central model factory (ERSR-style net_factory)."""

from networks.himamba import HiMamba


def net_factory(net_type='himamba', **kwargs):
	if net_type == 'himamba':
		return HiMamba(**kwargs)
	raise ValueError(f'Unknown net_type: {net_type}. Expected "himamba".')
