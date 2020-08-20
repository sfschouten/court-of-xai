local batch_size = 128;

{
  "dataset_reader": {
    "type": "quora_paraphrase",
    "tokenizer": {
      "type" : "pretrained_transformer",
      "model_name" : "distilbert-base-uncased",
      "add_special_tokens" : false
    },
    "token_indexers": { "tokens" : {
      "type" : "pretrained_transformer",
      "model_name" : "distilbert-base-uncased",
    }},
    "combine_input_fields" : true,
  },
  "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/QUORA/quora_train.tsv"]),
  "test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/QUORA/quora_train.tsv"]),
  "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/QUORA/quora_train.tsv"]),
  "evaluate_on_test": true,
  "model": {
    "type": "distilbert_sequence_classification_from_huggingface",
    "model_name": "distilbert-base-uncased",
    "ffn_activation": "gelu",
    "ffn_dropout": 0.2,
    "attention": {
        "type": "multihead_self",
        "n_heads": 12, 
        "dim": 768,
        "activation_function": {
            "type": "uniform"
        },
        "dropout": 0.2
    },
    "num_labels": 3,
    "seq_classif_dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2.0e-5
    },
  }
}
