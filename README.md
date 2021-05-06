# Court of XAI

## Introduction

Code to reproduce the results reported in the paper: "Order in the Court: Explainable AI Methods Prone to Disagreement". Our framework can be used to design and run additional experiments to calculate different correlation metrics between feature additive explanation methods. Currently, we assume you are interested in comparing at least one attention-based explanation method.

## Setup

### Installation

Prepare a Python virtual environment and install the necessary packages.

```shell
python3 -m venv v-xai-court
source v-xai-court/bin/activate
pip install --upgrade pip
pip install torch torchtext
pip install -r requirements.txt
python -m spacy download en
```

### Datasets

Datasets are stored in the `datasets/` folder. We include the IMDb and SST-2 datasets with the same splits and pre-processing steps as used in [(Jain and Wallace 2019)](https://arxiv.org/abs/1902.10186). To download the other datasets:

- [Quora](https://drive.google.com/file/d/12b-cq6D45U5c-McPoq2wsFjzs6QduY_y/view?usp=sharing)
  - Our split (80/10/10), question pairs with combined word count greater than 200 were removed.
- [SNLI](https://nlp.stanford.edu/projects/snli/)
- [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)
  - Get the test set from [here](https://cims.nyu.edu/~sbowman/xnli/) and run `scripts/extract_mnli_test_en.py`

## Reproducing our Paper

We have implemented a custom [AllenNLP](https://github.com/allenai/allennlp/) command to run our experiments. We define each experiment as consisting of four variables:

- The dataset (IMDb, SST-2, Quora Question Pair, SNLI, MultiNLI)
- The model (BiLSTM, DistilBERT)
- The attention mechanism (additive (tanh), self-attention)
- The attention activation function (softmax, uniform, etc...)

Thus the experiment files are named `{dataset}_{model}_{attention_mechanism}_{activation_function}.jsonnet`. Our experiments are located in the `experiments/` folder.

Since we train three independently-seeded models PER experiment, it may take several days to run all of our experiments. A GPU with CUDA and 16GB of memory is **strongly** recommended.

To run an experiment simply call our custom command with the path to the experiment file. For example:

```shell
allennlp attn-experiment experiments/sst_distilbert_self_softmax.jsonnet
```

By default, the generated artifacts are available in the `outputs` directory. This includes the models, their configurations, and their metrics.
When the experiments finish, a .csv summary of the correlation results in available in `outputs/{experiment_file_name}/summary.csv`. We used these summary files to generate our tables.

## Running your own Experiment

Since our code uses AllenNLP, you can easily add a new `Jsonnet` experiment file to the `experiments` directory.

We currently support the following components (see the existing experiment files for examples on how to use them):

- Tasks/Datasets
  - Single Sequence: Binary Sentiment Classification
    - IMDb Movie Reviews
    - Stanford Sentiment Treebank
  - Pair Sequence: Natural Language Inference
    - Quora Question Pairs
    - SNLI
    - MultiNLI
- Models
  - BiLSTM with (tanh) additive attention
  - [DistilBERT](https://arxiv.org/abs/1910.01108)
- Attention activation functions
  - Softmax
  - Uniform
  - [Sparsemax](https://arxiv.org/pdf/1602.02068.pdf)
  - [Alpha Entmax](https://www.aclweb.org/anthology/P19-1146) (alpha = 1.5 or learned)
- Attention aggregation and analysis methods:
  - Average: for the Transformer, averages attention across layers, heads, and max pools across the last dimension of attention matrix
  - [Attention Flow](https://www.aclweb.org/anthology/2020.acl-main.385/): for the Transformer
  - [Attention Rollout](https://www.aclweb.org/anthology/2020.acl-main.385/): for the Transformer
  - [Attention Norms](https://www.aclweb.org/anthology/2020.emnlp-main.574): as an additional analysis method for any attention mechanism
- Additive Feature Importance Methods (from [Captum](https://captum.ai/))
  - [LIME](https://arxiv.org/abs/1602.04938)
  - Feature Ablation
  - [Integrated Gradients](https://arxiv.org/abs/1703.01365)
  - [DeepLIFT](https://arxiv.org/abs/1704.02685)
  - [Gradient SHAP](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
  - [Deep SHAP]((http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf))
  - Attention (see previous)
- Correlation metrics
  - Kendall-tau
  - [Top-k Kendall-tau](https://www.researchgate.net/publication/2537159_Comparing_Top_k_Lists). This is a custom implementation where k can be a fixed number of tokens, variable percentage, or non-zero attribution values.
  - [Weighted Kendall-tau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html)
  - Yilmaz tauAP where ties are allowed. Taken from [Pyircor](https://github.com/eldrin/pyircor)
  - Spearman-r
  - Pearson-rho

## Citation

```bibtex

```
