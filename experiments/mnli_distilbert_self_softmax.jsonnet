local batch_size = 128;

{
    "dataset_reader": {
        "type": "snli",
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
    "train_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/MNLI/multinli_1.0_train.jsonl"]),
    #"test_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/MNLI/multinli_0.9_test_matched_unlabeled.jsonl"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "ane_research/datasets/MNLI/multinli_1.0_dev_matched.jsonl"]),
    "evaluate_on_test": false,
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
                        "type": "softmax"
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
                "type": "kendall_top_k_average_length"
            }
        ],
        "dataset": "MNLI",
        "model": "DistilBERT",
        "compatibility_function": "Self",
        "activation_function": "Softmax",
        "batch_size": batch_size,
        "cuda_device": 0
    }
}
