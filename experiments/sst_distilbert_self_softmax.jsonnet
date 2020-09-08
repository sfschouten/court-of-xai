local batch_size = 64;
local transformer_model = "distilbert-base-uncased";
{
    "dataset_reader": {
        "type": "sst_tokens",
        "granularity": "2-class",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model
        }
    },
    "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/train.txt"]),
    "test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/test.txt"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/SST/dev.txt"]),
    "evaluate_on_test": false,
    "model": {
        "type": "distilbert_sequence_classification_from_huggingface",
        "model_name": transformer_model,
        "ffn_activation": "gelu",
        "ffn_dropout": 0.2,
        "attention": {
            "type": "multihead_self",
            "n_heads": 12,
            "dim": 768,
            "activation_function": {
                "type": "softmax"
            },
            "dropout": 0.2
        },
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
        "num_epochs": 40,
        "patience": 5,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        }
    },
    "attention_experiment": {
        "feature_importance_measures": [
            {
                "type": "leave-one-out"
            },
            // {
            //     "type": "lime",
            //     "num_samples": 250
            // },
            {
                "type": "captum",
                "captum": "captum-integrated-gradients"
            },
            {
                "type": "captum",
                "captum": "captum-deepliftshap"
            },
            {
                "type": "captum",
                "captum": "captum-gradientshap"
            },
            {
                "type": "captum",
                "captum": "captum-deeplift"
            }
        ],
        "correlation_measures": [
            {
                "type": "kendall_tau"
            },
            {
                "type": "kendall_top_k_average_length"
            }
        ],
        "dataset": "SST",
        "model": "DistilBERT",
        "compatibility_function": "Self",
        "activation_function": "Softmax",
        "batch_size": batch_size,
        "cuda_device": -1
    }
}
