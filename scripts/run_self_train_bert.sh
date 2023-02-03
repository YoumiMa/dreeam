NAME=$1
TEACHER_DIR=$2
LAMBDA=$3
SEED=$4

python run.py --do_train \
--data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--train_file train_distant.json \
--dev_file dev.json \
--teacher_sig_path ${TEACHER_DIR} \
--save_path ${NAME} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 2 \
--evaluation_steps 5000 \
--num_labels 4 \
--lr_transformer 3e-5 \
--max_grad_norm 5.0 \
--evi_thresh 0.2 \
--attn_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 2.0 \
--seed ${SEED} \
--num_class 97 