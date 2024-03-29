local batch_size = 64;
local alpha_param_re = "^.*attention\\.activation\\.alpha";
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
    "train_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_train.tsv"]),
    "test_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_test.tsv"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "datasets/QUORA/quora_dev.tsv"]),
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
            "lr": 2.0e-5,
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
                "type": "captum",
                "captum": {
                    "type": "captum-lime",
                    "attribute_args": {
                        "n_samples": 1000
                    }
                }
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
        "dataset": "Quora",
        "model": "DistilBERT",
        "compatibility_function": "Self",
        "activation_function": "Entmax",
        "batch_size": batch_size,
        "nr_instances": 500
    }
}
