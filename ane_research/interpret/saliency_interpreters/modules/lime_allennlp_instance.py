"""
Functions for explaining text classifiers using AllenNLP.
"""
from functools import partial
import itertools
import json
import re

import torch
import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

from lime import explanation
from lime import lime_base

from allennlp.data.fields import TextField

class AllenNLPInstanceDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, instance):
        """Initializer.

        Args:
            instance: TODO
        """
        self.instance = instance

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id,weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            raise NotImplementedError("Word positions not implemented for lime with AllenNLP.")
        else:
            exp = None
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id,weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        #TODO
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)
        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]) for x in exp]
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class LimeAllenNLPInstanceExplainer(object):
    """Explains text classifiers within the AllenNLP framework.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.char_level = char_level

    def explain_instance(self,
                         instance,
                         model,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            instance: TODO 
            model: TODO classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        domain_mapper = AllenNLPInstanceDomainMapper(instance)
        data, yss, distances = self.__data_labels_distances(
            instance, model, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                instance,
                                model,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the model. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            instance: instance to be explained,
            model: 
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.

        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        vocab = model.vocab
        pad_token = vocab.get_token_index(vocab._padding_token)
        
        inverse_data = [ instance ]
        
        doc_sizes = {}
        samples = {}
        data = {}
        features_ranges = {}
        for key,field in instance.items():
            if not isinstance(field, TextField):
                continue

            doc_size = len(field)
            doc_sizes[key] = doc_size 
           
            # TODO verify that doc_size is fine (was doc_size + 1), got nan 
            # values for cases where all tokens were removed.
            samples[key] = self.random_state.randint(1, doc_size, num_samples - 1)
            data[key] = np.ones((num_samples, doc_size))
            data[key][0] = np.ones(doc_size)
            features_ranges[key] = range(doc_size)

        
        for i in range(1, num_samples):
            removed = instance.duplicate()

            for field_key in removed.keys():
                field = removed[field_key]
                if not isinstance(field, TextField):
                    continue

                size = samples[field_key][i-1]
                features_range = features_ranges[field_key]
                inactive = self.random_state.choice(features_range, size, replace=False)
                
                data[field_key][i, inactive] = 0
                
                for idxr, tokenlist in removed[field_key]._indexed_tokens.items():
                    for j in inactive:
                        for key in tokenlist.keys():
                            # TODO: this (use fields with 'token') is too brittle, 
                            # should make model/predictor implement interface that exposes the name of the relevant field.
                            if 'token' in key:
                                tokenlist[key][j] = pad_token

            inverse_data.append(removed)

        data = np.concatenate(tuple(data[key] for key in reversed(list(data.keys()))), axis=1)

        BATCH_SIZE = 16
        batches = (itertools.islice(inverse_data, x, x+BATCH_SIZE) for x in range(0, len(inverse_data), BATCH_SIZE))
        results = []
        for idx, batch in enumerate(batches):
            batch = list(batch)
            results.extend(model.forward_on_instances(batch))

        class_probs = np.asarray([ result['class_probabilities'] for result in results ])
        distances = distance_fn(sp.sparse.csr_matrix(data))

        return data, class_probs, distances
