# fact-ai-2020
 Fairness, Accountability, Confidentiality and Transparency in AI Course Assignment, January 2020

## Setup

```shell
  python3 -m venv v-ane-research
  source v-ane-research/bin/activate
  pip install --upgrade pip
  pip install torch torchtext
  pip install -r requirements.txt
```

## Running Experiments

### Run Binary Sentiment Classification on SST Dataset

From root:

```shell
  allennlp train ane_research/experiments/sst_binary_sentiment.jsonnet -s outputs --include-package ane_research
```
