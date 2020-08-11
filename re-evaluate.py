#!/usr/bin/env python3

import os
import argparse

from ane_research.utils.experiments import get_most_recent_trained_model_paths
from ane_research.evaluate import Evaluator
from ane_research.config import Config

# import the models from our and official package.
from allennlp.common import util as common_util
common_util.import_module_and_submodules(Config.package_name)
common_util.import_module_and_submodules('allennlp_models')

#######
parser = argparse.ArgumentParser()
parser.add_argument('experiment_keyword', type=str, default='')
args = parser.parse_args()
#######

recent_models = get_most_recent_trained_model_paths()
for idx, recent_model in enumerate(recent_models):

    _, recent_model_file = os.path.split(recent_model)
    experiment_id, experiment_name = os.path.splitext(recent_model_file)

    if args.experiment_keyword not in recent_model:
        continue

    recent_model += '/model.tar.gz'

    print(f"Loading {recent_model}.")

    evaluator = Evaluator(model_path = recent_model, calculate_on_init = True, experiment_name=experiment_name)
    evaluator.generate_and_save_correlation_data_frames()
    evaluator.generate_and_save_correlation_graphs()

