import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union, Tuple, Generator

from allennlp.common import Registrable
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ane_research.common.correlation_measures import CorrelationMeasure
import ane_research.common.plotting as plotting
import ane_research.common.utils as utils
from ane_research.config import Config
from ane_research.experiments.attention_correlation.trial import AttentionCorrelationTrial


class AttentionCorrelationExperiment(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        dataset: str,
        model: str,
        compatibility_function: str,
        activation_function: str,
        serialization_dir: PathLike,
        feature_importance_measures: List[str],
        attention_measures: List[str],
        correlation_measures: List[CorrelationMeasure],
        feature_importance_results: pd.DataFrame,
        correlation_results: pd.DataFrame
    ):
        self.dataset = dataset
        self.model = model
        self.compatibility_function = compatibility_function
        self.activation_function = activation_function
        self.serialization_dir = serialization_dir
        self.feature_importance_measures = feature_importance_measures
        self.attention_measures = attention_measures

        self.correlation_measures = correlation_measures
        self.feature_importance = feature_importance_results
        self.correlation = correlation_results

        self.logger =  logging.getLogger(Config.logger_name)

        self.graph_path = utils.ensure_dir(f"{self.serialization_dir}/graphs/")

        plotting.set_styles()
        self.plot_subtitle = self._get_plot_subtitle()

    def _get_plot_subtitle(self):
        return (\
            f"Dataset: {self.dataset} • Model: {self.model} • Attention: "\
            f"{self.compatibility_function} with {self.activation_function}"\
        )

    def _plot_correlation_density(self):
        group_by = ['instance_id', 'feature_importance_measure', 'attention_measure']
        avg_across_trials = self.correlation.groupby(group_by, as_index=False)
        avg_across_trials = avg_across_trials.agg({cm.id: ['mean'] for cm in self.correlation_measures})
        avg_across_trials.columns = group_by + [cm.id for cm in self.correlation_measures]
        for attention_measure in self.attention_measures:
            correlation = avg_across_trials[avg_across_trials['attention_measure'] == attention_measure]
            for cm in self.correlation_measures:
                fig, ax = plt.subplots()
                plotting.multi_univariate_kde(
                    frame=correlation,
                    data_col=cm.id,
                    label_col='feature_importance_measure',
                    ax=ax,
                    clip=(-1, 1)
                )
                plotting.annotate(
                    fig=fig,
                    ax=ax,
                    xlabel=f'Correlation ({cm.id})',
                    ylabel='Density',
                    title=f"Correlation with Attention {attention_measure.split('_')[1].capitalize()}",
                    subtitle=self.plot_subtitle
                )
                plotting.save_figure(self.graph_path, f'{cm.id}_{attention_measure}')

    def _plot_correlation_by_length(self):
        self.correlation['instance_length'] = self.correlation['instance_text'].apply(lambda t: sum(len(l) for l in t))
        for attention_measure in self.attention_measures:
            corr_by_attn_measure = self.correlation[self.correlation['attention_measure'] == attention_measure]
            for cm in self.correlation_measures:
                fig, ax = plt.subplots()
                ax = sns.lineplot(
                    x='instance_length',
                    y=cm.id,
                    data=corr_by_attn_measure,
                    style='feature_importance_measure',
                    hue='feature_importance_measure'
                )
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[1:], labels=labels[1:])
                plotting.annotate(
                    fig=fig,
                    ax=ax,
                    xlabel=f'Instance Length (tokens)',
                    ylabel=cm.id,
                    title=f"Correlation with Attention {attention_measure.split('_')[1].capitalize()}",
                    subtitle=self.plot_subtitle
                )
                plotting.save_figure(self.graph_path, f'correlation_by_length_{cm.id}_{attention_measure}')

    def _plot_correlation_by_class(self):
        for attention_measure in self.attention_measures:
            corr_by_attn_measure = self.correlation[self.correlation['attention_measure'] == attention_measure]
            for cm in self.correlation_measures:
                fig, ax = plt.subplots()
                sns.barplot(
                    x='predicted',
                    y=cm.id,
                    data=corr_by_attn_measure,
                    hue='feature_importance_measure'
                )
                plotting.annotate(
                    fig=fig,
                    ax=ax,
                    xlabel=f'Class',
                    ylabel=cm.id,
                    title=f"Correlation with Attention {attention_measure.split('_')[1].capitalize()}",
                    subtitle=self.plot_subtitle
                )
                plotting.save_figure(self.graph_path, f'correlation_by_class_{cm.id}_{attention_measure}')

    # TODO: cleanup and remove the hardcoded correlation fields!
    def _summarize(self):
        summary = self.correlation
        summary['accuracy'] = summary['predicted'] == summary['actual']
        summary['accuracy'] = summary['accuracy'].astype(float)

        if "p_val" in summary.columns:
            summary['p_val'] = summary['p_val'].apply(lambda p: p < 0.05)
        summary = summary.drop(['predicted', 'instance_fields', 'instance_text'], axis=1)

        # Average for each trial
        summary = summary.groupby(['feature_importance_measure', 'attention_measure', 'seed', 'actual'], as_index = False)
        agg_by_seed = {f: ['mean'] for cm in self.correlation_measures for f in cm.fields}
        agg_by_seed['instance_id'] = ['count']
        agg_by_seed['accuracy'] = ['mean']
        agg_by_seed_columns = [
            'feature_importance_measure',
            'attention_measure',
            'seed',
            'class'
        ]
        agg_by_seed_columns.extend([f for cm in self.correlation_measures for f in cm.fields])
        agg_by_seed_columns.extend(['instance_id', 'accuracy'])
        summary = summary.agg(agg_by_seed)
        summary.columns = agg_by_seed_columns

        # Average, std across trials
        summary = summary.groupby(['feature_importance_measure', 'attention_measure', 'class'],  as_index=False)
        agg_by_class = {f: ['mean', 'std'] for cm in self.correlation_measures for f in cm.fields}
        agg_by_class['instance_id'] = ['mean']
        agg_by_class['accuracy'] = ['mean', 'std']
        agg_by_class_columns = [
            'feature_importance_measure',
            'attention_measure',
            'class'
        ]
        agg_by_class_columns.extend([f'{f}_{agg}' for cm in self.correlation_measures for f in cm.fields for agg in ['mean', 'std']])
        agg_by_class_columns.extend(['instance_count', 'accuracy_mean', 'accuracy_std'])
        summary = summary.agg(agg_by_class)
        summary.columns = agg_by_class_columns
        if "p_val_mean" in summary.columns:
            summary = summary.rename(columns={"p_val_mean": "fraction_significant_mean"})
        if "p_val_std" in summary.columns:
            summary = summary.rename(columns={"p_val_std": "fraction_significant_std"})
        if "k_average_length_mean" in summary.columns:
            summary = summary.rename(columns={"k_average_length_mean": "k_average_length"})
        if "k_average_length_std" in summary.columns:
            summary = summary.drop(["k_average_length_std"], axis=1)
        self.summary = summary

    def generate_artifacts(self):

        self._summarize()

        # Generate Frames
        utils.write_frame(self.correlation, self.serialization_dir, 'correlations_all')
        utils.write_frame(self.correlation, self.serialization_dir, 'feature_importance_all')
        utils.write_frame(self.summary, self.serialization_dir, 'results')

        # Generate Figures
        self._plot_correlation_density()
        self._plot_correlation_by_length()
        self._plot_correlation_by_class()

        # TODO: potentially generate heatmaps over instance text

        # TODO: Generate a MarkDown or HTML report


    @classmethod
    def from_completed_trials(
        cls,
        dataset: str,
        model: str,
        compatibility_function: str,
        activation_function: str,
        serialization_dir: str,
        trials: List[AttentionCorrelationTrial]
    ):
        # TODO: handle case if different trials have different measures? Perhaps take intersection?
        feature_importance_measures = [fi.id for fi in trials[0].feature_importance_interpreters]
        attention_measures = [ai.id for ai in trials[0].attention_interpreters]
        correlation_measures = trials[0].correlation_measures

        feature_importance_results = pd.concat([trial.feature_importance_results for trial in trials])
        correlation_results = pd.concat([trial.correlation_results for trial in trials])

        return cls(
            dataset=dataset,
            model=model,
            compatibility_function=compatibility_function,
            activation_function=activation_function,
            serialization_dir=serialization_dir,
            feature_importance_measures=feature_importance_measures,
            attention_measures=attention_measures,
            correlation_measures=correlation_measures,
            feature_importance_results=feature_importance_results,
            correlation_results=correlation_results
        )

AttentionCorrelationExperiment.register("default", constructor="from_completed_trials")(AttentionCorrelationExperiment)
