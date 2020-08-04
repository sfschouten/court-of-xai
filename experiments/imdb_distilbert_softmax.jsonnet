local batch_size = 64;

{
  "dataset_reader": {
    "type": "imdb_csv",
    "max_review_length": 100,
    "pretrained_tokenizer": "distilbert-base-uncased"
  },
  "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/train.csv"]),
  "test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/test.csv"]),
  "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/dev.csv"]),
  "evaluate_on_test": true,
  "model": {
    "type": "distilbert_sequence_classification_from_huggingface",
    "model_name": "distilbert-base-uncased",
    "ffn_activation": "gelu",
    "attention_activation": "softmax",
    "attention_dropout": 0.2,
    "num_labels": 2,
    "seq_classif_dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size
    }
  },
  "trainer": {
    "num_epochs": 10,
    "patience": 5,
    "cuda_device": -1,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1.0e-5
    },
  }
}
