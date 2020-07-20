from ane_research.utils.experiments import run_experiment, run_all_experiments_in_dir

import argparse
parser = argparse.ArgumentParser(description='run one (or all) experiments')
parser.add_argument('--experiment_path', type=str, required=False)
parser.add_argument('--recover', action='store_true', required=False)

args, _ = parser.parse_known_args()

if args.experiment_path:
  run_experiment(args.experiment_path, recover = args.recover)
else:
  run_all_experiments_in_dir('experiments')
