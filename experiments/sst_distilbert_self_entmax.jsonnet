local batch_size = 64;
local alpha_param_re = "^.*attention\\.activation\\.alpha";
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
    "evaluate_on_test": true,
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
        "attention_aggregator_methods": [
            "roll",
            "flow"
        ],
        "feature_importance_measures": [
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
                    0.25,
                    0.5,
                    1.0,
                ],
            },
            {
                "type": "kendall_top_k_fixed",
                "fixed_top_k": [
                    5,
                    10
                ],
            }
        ],
        "dataset": "SST",
        "model": "DistilBERT",
        "compatibility_function": "Self",
        "activation_function": "Entmax",
        "batch_size": batch_size,
        "nr_instances": 500,
        "cuda_device": 0
    }
}
