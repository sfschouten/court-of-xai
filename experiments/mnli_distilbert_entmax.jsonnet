local batch_size = 128;
local alpha_param_re = "^.*attention\\.activation\\.alpha";
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
    }
}