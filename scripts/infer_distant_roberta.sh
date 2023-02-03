NAME=$1
LOAD_DIR=$2


python run.py --data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--display_name  ${NAME} \
--load_path ${LOAD_DIR} \
--eval_mode single \
--test_file train_distant.json \
--test_batch_size 4 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97 \
--save_attn \
