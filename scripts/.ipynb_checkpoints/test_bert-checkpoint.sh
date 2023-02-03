TYPE=$1
LAMBDA=$2
SEED=$3

NAME=test_${TYPE}_lambda${LAMBDA}

python train.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--load_path repr_last_3_layers_valid_r_lambda0.1/2022-09-29_18:30:33.841193/ \
--eval_mode single \
--pred_file test_result.json \
--thresh -1.1540794372558594  \
--test_file test.json \
--save_path ${NAME} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 5e-5 \
--max_grad_norm 1.0 \
--evi_thresh 0.2 \
--seed ${SEED} \
--num_class 97 

python train.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--load_path repr_last_3_layers_valid_r_lambda0.1/2022-09-29_18:30:33.841193/  \
--eval_mode multi \
--pred_file test_result.json \
--thresh -1.1540794372558594 \
--test_file test.json \
--save_path ${NAME} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 5e-5 \
--max_grad_norm 1.0 \
--evi_thresh 0.2 \
--seed ${SEED} \
--num_class 97 
