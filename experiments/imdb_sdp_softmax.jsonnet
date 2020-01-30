local encoder_hidden_size = 128;
local embedding_dim = 300;
local batch_size = 64;

{
  "dataset_reader": {
    "type": "imdb_csv",
  },
  "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/train.csv"]),
  "test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/test.csv"]),
  "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/IMDB/dev.csv"]),
  "evaluate_on_test": true,
  "model": {
    "type": "jain_wallace_attention_binary_classifier",
    "word_embeddings": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
          "embedding_dim": embedding_dim,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": embedding_dim,
      "hidden_size": encoder_hidden_size,
      "num_layers": 1,
    },
    "decoder": {
      "input_dim": encoder_hidden_size * 2,
      "num_layers": 1,
      "hidden_dims": 1,
      "activations": ["linear"]
    },
    "attention": {
      "type": "additive_sdp",
      "hidden_size": encoder_hidden_size * 2,
      "activation_function": "softmax" 
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [['tokens', 'num_tokens']],
    "batch_size": batch_size
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "validation_metric": "+auc",
    "optimizer": {
      "type": "adam",
      "weight_decay": 1e-5,
      "amsgrad": true
    }
  }
}
