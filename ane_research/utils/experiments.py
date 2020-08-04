import fnmatch
import logging
import glob
import os
import subprocess
from typing import List

from datetime import datetime, timezone
from ane_research.config import Config
from ane_research.evaluators import JWAEDEvaluator

from allennlp.commands.train import train_model_from_file

class Experiment():
  ''''Helper class to make running experiments and generating csv files and images as easy as possible'''
  def __init__(self, experiment_file_path: str, recover: bool = False, outputs_path: str = 'outputs'):
    self.recover = recover

    self.experiment_file_path = experiment_file_path
    _, self.experiment_file_name = os.path.split(self.experiment_file_path)
    self.experiment_name, extension = os.path.splitext(self.experiment_file_name)

    if recover:
      recent = get_most_recent_trained_model_paths()
      recoveree = next(x for x in recent if x.startswith(f"{outputs_path}/{self.experiment_name}"))
      self.out_path = recoveree
    else:
      timestamp = int(datetime.now(timezone.utc).timestamp())
      self.out_path = f'{outputs_path}/{self.experiment_name}/{timestamp}'
    
    self.model_path = self.out_path + '/model.tar.gz'
 
    self.dataset_name, self.attention_type, self.attention_activation_function = self.experiment_name.split('_')

    # import the models from our and official package.
    from allennlp.common import util as common_util
    common_util.import_module_and_submodules(Config.package_name)
    common_util.import_module_and_submodules('allennlp_models')


  def train(self) -> None:
    train_model_from_file(
      self.experiment_file_path,
      self.out_path,
      recover = self.recover
    )

  def evaluate(self) -> None:
    self.evaluator = JWAEDEvaluator(model_path = self.model_path, calculate_on_init=True)

  def generate_and_save_artifacts(self) ->  None:
    self.evaluator.generate_and_save_correlation_data_frames()
    self.evaluator.generate_and_save_correlation_graphs()

  def get_figure_paths(self) -> List[str]:
    figure_paths = []
    for file in os.listdir(self.evaluator.graph_path):
      if fnmatch.fnmatch(file, '*.png'):
        figure_paths.append(self.evaluator.graph_path + '/' + file)
    return figure_paths

def run_all_experiments_in_dir(dir_path: str = 'experiments') -> List[Experiment]:
  '''Run all AllenNLP experiments in the given directory'''
  experiments = []
  for file in os.listdir(dir_path):
    if fnmatch.fnmatch(file, '*.jsonnet'):
      experiment_path = f'{dir_path}/{file}'
      experiment = run_experiment(experiment_path)
      experiments.append(experiment)
  return experiments

def run_experiment(experiment_path: str, recover: bool = False):
  experiment = Experiment(experiment_path, recover = recover)
  experiment.train()
  experiment.evaluate()
  experiment.generate_and_save_artifacts()
  return experiment

def get_most_recent_trained_model_paths(outputs_path: str = 'outputs') -> List[str]:
  '''Retrieve the filepaths of the most recent model run of each unique experiment
    This function assumes models are located in `outputs_path/experiment_name/timestamp`
    See the `run_experiment function`

    Args
      - outputs_path: (str): Path to where models are generated and saved.
    Returns
      - List[str]: List of paths to the most recent model run of each unique experiment
  '''

  most_recent_model_paths = []
  possible_matches = glob.glob(f'{outputs_path}/**/[0-9]*')
  output_groups = {}
  for possible_match in possible_matches:
    splits = possible_match.split('/')
    splits.pop()
    dir_prefix = '/'.join(splits)
    if dir_prefix not in output_groups.keys():
      output_groups[dir_prefix] = [possible_match]
    else:
      output_groups[dir_prefix].append(possible_match)

  for group in output_groups.keys():
    most_recent_model_paths.append(max(output_groups[group], key=lambda s: datetime.fromtimestamp(int(s.split('/')[-1]))))

  return most_recent_model_paths
