import argparse
from pathlib import Path

from sfw_brood.preprocessing import discover_training_data, prepare_training

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('out_path', type = str)
	arg_parser.add_argument('-w', '--work-dir', type = str)
	arg_parser.add_argument('-o', '--overlap', type = float, default = 0.0)
	arg_parser.add_argument('-s', '--sample-len', type = float, default = 2.0)
	args = arg_parser.parse_args()

	out_dir = f's{args.sample_len}-o{args.overlap}'

	dataset = discover_training_data(args.data_path)
	bs_df, ba_df = prepare_training(
		dataset, work_dir = f'{args.work_dir}/{out_dir}',
		slice_duration_sec = args.sample_len, overlap_sec = args.overlap
	)

	out_path = Path(args.out_path).joinpath(out_dir)
	out_path.mkdir(exist_ok = True, parents = True)

	bs_df.to_csv(out_path.joinpath('brood-size.csv'))
	ba_df.to_csv(out_path.joinpath('brood-age.csv'))
