local batch_size = 64;
local alpha_param_re = "^.*attention\\.activation\\.alpha";
{
    "dataset_reader": {
        "type": "imdb_csv",
        "max_review_length": 240,
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
        "ffn_dropout": 0.2,
        "attention": {
            "type": "multihead_self",
            "n_heads": 12,
            "dim": 768,
            "activation_function": {
                "type": "entmax-alpha",
                "alpha": 1.5
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
            "lr": 1.0e-5,
            "parameter_groups": [
                [[alpha_param_re], {"lr": 1.0e-3}]
            ]
        },
        "epoch_callbacks" : [
            {
                "type": "print-parameter",
                "param_re" : alpha_param_re 
            }
        ]
    },
    "attention_experiment": {
        "feature_importance_measures": [
            {
                "type": "leave-one-out"
            },
            {
                "type": "lime",
                "num_samples": 250
            },
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
                "type": "spearman_rho"
            },
            {
                "type": "pearson_r"
            },
            {
                "type": "kendall_top_k_variable",
                "percent_top_k": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ],
            },
            {
                "type": "kendall_top_k_fixed",
                "fixed_top_k": [
                    1,
                    3,
                    5,
                    10
                ],
            }
        ],
        "dataset": "IMDb",
        "model": "DistilBERT",
        "compatibility_function": "Self",
        "activation_function": "Entmax",
        "batch_size": batch_size,
        "nr_instances": 500,
        "cuda_device": 0
    }
}
