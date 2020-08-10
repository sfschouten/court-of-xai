## TODO
- abstract
- new name for ane\_research

# Sparse Attention is Consistent with Feature Importance

## Abstract

## Authors

- Michael J. Neely
- Stefan F. Schouten

## Datasets
- SNLI (https://nlp.stanford.edu/projects/snli/)
- IMDb
- SST-2

## Setup

Prepare a Python virtual environment and install the necessary packages.

```shell
 python3 -m venv v-ane-research
 source v-ane-research/bin/activate
 pip install torch torchvision torchtext
 pip install -r requirements.txt
 python -m spacy download en
```

## Reproducing our Experiments

From the base directory:

### Option 1

Run all AllenNLP `Jsonnet` files in the `experiments` directory:

```sh
python run_experiment.py
```

### Option 2

Run a specific `Jsonnet` experiment

```sh
python run_experiment.py --experiment_path experiments/sst_tanh_sparsemax.jsonnet
```

### Option 3

Open the `replicate_paper.ipynb` iPython notebook with your preferred Jupyter server.

### Option 4

Instantiate a new `Experiment` class in your own Python file.

## Viewing the Results

Trained models and experiment results are located in the `outputs` directory, indexed by experiment file name and timestamp. You can view the correlation graphs as `.png`, `.svg`, or `.pdf` images in the `graphs` sub-folder. You can view the correlation statistics in the `correlation` sub-folder.

## Extending our Research

Since our code uses AllenNLP, you can easily add a new `Jsonnet` experiment file to the `experiments` directory.

## Linting
Simply run `python setup.py lint`

## Citation

```bibtex
@misc{jain2019attention,
    title={Attention is not Explanation},
    author={Sarthak Jain and Byron C. Wallace},
    year={2019},
    eprint={1902.10186},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
