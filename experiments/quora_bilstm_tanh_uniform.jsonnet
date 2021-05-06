local encoder_hidden_size = 128;
local embedding_dim = 300;

{
    "dataset_reader": {
        "type": "quora_paraphrase"
    },
    "train_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_train.tsv"]),
    "test_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_test.tsv"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_dev.tsv"]),
    "evaluate_on_test": true,
    "model": {
        "type": "pair_sequence_classifier",
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
            "input_dim": encoder_hidden_size * 8,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": ["linear"]
        },
        "attention": {
            "type": "additive_basic",
            "hidden_size": encoder_hidden_size * 2,
            "activation_function": {
                "type": "uniform"
            }
        },
        "field_names": ["premise", "hypothesis"],
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 1024
        }
    },
    "trainer": {
        "num_epochs": 40,
        "patience": 5,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam",
            "weight_decay": 1e-5,
            "amsgrad": true
        }
    }
}
