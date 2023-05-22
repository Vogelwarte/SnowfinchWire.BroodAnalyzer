from pathlib import Path
from typing import Union, Optional, Tuple


def make_dataset_path(
		base_path: Union[Path, str], data_config_id: str, target: str,
		samples_per_class: Union[int, str], age_range: Optional[Tuple[float, float]] = None
) -> Path:
	if age_range:
		low, high = age_range
		age_info = f'-age-{low}-{high}'
	else:
		age_info = ''

	return Path(base_path) \
		.joinpath('nemo') \
		.joinpath(data_config_id) \
		.joinpath(f'spc-{samples_per_class}{age_info}') \
		.joinpath(target)
