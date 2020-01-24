{
  "dataset_reader": {
    "type": "sst_tokens",
    "granularity": "2-class"
  },
  "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/train.txt"]),
  "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/dev.txt"]),
  "test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/test.txt"]),
  "model": {
    "type": "jain_wallace_attention_binary_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        "embedding_dim": 300,
        "trainable": false
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 300,
      "hidden_size": 128,
      "num_layers": 1,
    },
    "decoder": {
      "input_dim": 256,
      "num_layers": 1,
      "hidden_dims": 1,
      "activations": ["linear"]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [['tokens', 'num_tokens']],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    # "grad_clipping": 5.0,
    "validation_metric": "+auc",
    "optimizer": {
      "type": "adam",
      "weight_decay": 1e-5,
      "amsgrad": true
    }
  }
}
