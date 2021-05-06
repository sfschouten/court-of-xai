"""
Train one model per seed and calculate the correlation between its suitable aggregated attention analysis
methods and the feature importance measures defined in the experiment configuration.
"""
import argparse
from copy import deepcopy
import inspect
import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union, Tuple

from allennlp.common import Params
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model
from overrides import overrides

import xai_court.common.utils as utils
from xai_court.config import Config
from xai_court.experiments.attention_correlation.experiment import AttentionCorrelationExperiment
from xai_court.experiments.attention_correlation.trial import AttentionCorrelationTrial


logger = logging.getLogger(Config.logger_name)


@Subcommand.register("attn-experiment")
class RunAttentionExperiment(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = (\
            "Train one model per seed and calculate the correlation between its "\
            "suitable aggregated attention analysis methods and the feature importance "\
            "measures defined in the experiment configuration. The results from all trials, "\
            "as well as the aggregated results and figures, are saved in the specified "\
            "output directory."\
        )

        subparser = parser.add_parser(self.name, description=description, help="FILL IN")

        subparser.add_argument(
            "experiment_path",
            type=str,
            help="path to the .jsonnet file describing the attention experiment."
        )

        subparser.add_argument(
            "-o",
            "--serialization-dir",
            type=str,
            default=None,
            help=(\
                "base directory in which to save models, their logs and, and their results. "\
                "Each trial is saved in a seed_x sub-directory."\
            )
        )

        subparser.add_argument(
            "-s",
            "--seeds",
            type=int,
            nargs="+",
            default=Config.seeds,
            help="One trial will be conducted per random seed"
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir."
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory, including ALL trained models and dataframes if it exists."
        )

        subparser.add_argument(
            "--train-only",
            action="store_true",
            required=False,
            help="Only train the model, do not calculate feature importance measures or correlations."
        )

        subparser.add_argument(
            "--re-calculate",
            choices=["correlation", "both"],
            default=None,
            help=(\
                "If a model has already been trained, re-calculate the feature importance measures and their "\
                "correlations ('both') or just the correlations ('correlation')."\
            )
        )

        subparser.add_argument(
            "--debug",
            action="store_true",
            required=False,
            help=(\
                "debugging mode: reduce max instances to 100, only train for one epoch, "\
                "and limit calculations to 10 instances."\
            )
        )

        subparser.set_defaults(func=run_attention_experiment_from_args)

        return subparser

def run_attention_experiment_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    run_attention_experiment_from_file(
        experiment_filename=args.experiment_path,
        serialization_dir=args.serialization_dir,
        seeds=args.seeds,
        recover=args.recover,
        force=args.force,
        debug=args.debug,
        train_only=args.train_only,
        re_calculate=args.re_calculate
    )

def run_attention_experiment_from_file(
    experiment_filename: PathLike,
    serialization_dir: Optional[Union[str, PathLike]] = None,
    seeds: List[int] = Config.seeds,
    recover: Optional[bool] = False,
    force: Optional[bool] = False,
    debug: Optional[bool] = False,
    train_only: Optional[bool] = False,
    re_calculate: Optional[str] = None
):
    """
    A wrapper around `run_attention_experiment` which loads the params from a file.
    """
    experiment_name = os.path.splitext(os.path.basename(experiment_filename))[0]

    if not serialization_dir:
        serialization_dir = os.path.join(Config.serialization_base_dir, experiment_name)

    params = Params.from_file(experiment_filename)

    if debug:
        params['dataset_reader']['max_instances'] = 100
        params['trainer']['num_epochs'] = 1
        params['attention_experiment']['nr_instances'] = 10

    run_attention_experiment(
        params=params,
        name=experiment_name,
        serialization_dir=serialization_dir,
        seeds=seeds,
        recover=recover,
        force=force,
        train_only=train_only,
        re_calculate=re_calculate
    )

def run_trial(
    trial_params: Params,
    train_params: Params,
    serialization_dir: PathLike,
    seed: int,
    recover: Optional[bool] = False,
    force: Optional[bool] = False,
    train_only: Optional[bool] = False,
    re_calculate: Optional[str] = None
) -> AttentionCorrelationTrial:

    _trial_params = deepcopy(trial_params)
    _train_params = deepcopy(train_params)

    test_data_path = _train_params['test_data_path']

    trial_dir = os.path.join(serialization_dir, f"seed_{seed}")

    should_train = any([force, not utils.model_already_trained(trial_dir), recover])

    if should_train:
        _train_params['random_seed'] = seed
        _train_params['numpy_seed'] = seed
        _train_params['pytorch_seed'] = seed

        train_model(
            params=_train_params,
            serialization_dir=trial_dir,
            recover=recover,
            force=force
        )

    if train_only:
        logger.info("'train-only' was specified. Finishing without calculating measures.")
        return

    attention_trial = AttentionCorrelationTrial.from_params(
        params=_trial_params,
        seed=seed,
        serialization_dir=trial_dir,
        test_data_path=test_data_path
    )

    recalc_fi = re_calculate == "both"
    recalc_corr = recalc_fi or re_calculate == "correlation"

    attention_trial.calculate_feature_importance(force=recalc_fi)
    attention_trial.calculate_correlation(force=recalc_corr)

    return attention_trial

def run_attention_experiment(
    params: Params,
    name: str,
    serialization_dir: PathLike,
    seeds: Optional[List[int]] = Config.seeds,
    recover: Optional[bool] = False,
    force: Optional[bool] = False,
    train_only: Optional[bool]= False,
    re_calculate: Optional[str] = None
):

    train_params = deepcopy(params)


    if "attention_experiment" not in train_params.params:
        logger.warn("Configuration file missing 'attention_experiment' parameters. Enabling train-only mode.")
        train_only = True
        trial_params = None
    else:
        additional_params = train_params.params.pop("attention_experiment")
        necessary_params_for_trial = inspect.getfullargspec(AttentionCorrelationTrial.from_partial_objects)[0]
        necessary_params_for_experiment = inspect.getfullargspec(AttentionCorrelationExperiment.from_completed_trials)[0]
        trial_params = Params({k: v for k, v in additional_params.items() if k in necessary_params_for_trial})
        experiment_params = Params({k: v for k, v in additional_params.items() if k in necessary_params_for_experiment})

    trials = []
    for seed in seeds:
        trials.append(
            run_trial(
                trial_params=trial_params,
                train_params=train_params,
                serialization_dir=serialization_dir,
                seed=seed,
                recover=recover,
                force=force,
                train_only=train_only,
                re_calculate=re_calculate
            )
        )

    logger.info("Finished training all models")

    if train_only:
        logger.info("'train-only' was specified. Finishing without aggregating trials.")
        return

    attention_experiment = AttentionCorrelationExperiment.from_params(
        params=experiment_params,
        serialization_dir=serialization_dir,
        trials=trials
    )

    attention_experiment.generate_artifacts()
