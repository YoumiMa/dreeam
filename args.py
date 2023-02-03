import argparse


def add_args(parser):
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--display_name", default=None, type=str)
    
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="", type=str)
    parser.add_argument("--pred_file", default="results.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--teacher_sig_path", default="", type=str)
    parser.add_argument("--save_attn", action="store_true", help="Whether store the evidence distribution or not")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--eval_mode", default="single", type=str,
                        choices=["single", "fushion"], 
                        help="Single-pass evaluation or evaluation with inference-stage fusion.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--max_sent_num", default=25, type=int,
                        help="Max number of sentences in each document.")
    parser.add_argument("--evi_thresh", default=0.2, type=float,
                        help="Evidence Threshold. ")
    parser.add_argument("--evi_lambda", default=0.1, type=float,
                        help="Weight of relation-agnostic evidence loss during training. ")
    parser.add_argument("--attn_lambda", default=1.0, type=float,
                        help="Weight of knowledge distillation loss for attentions during training. ")
    parser.add_argument("--lr_transformer", default=5e-5, type=float,
                        help="The initial learning rate for transformer.")
    parser.add_argument("--lr_added", default=1e-4, type=float,
                        help="The initial learning rate for added modules.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    return parser