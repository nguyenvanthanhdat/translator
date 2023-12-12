MODEL_NAME_OR_PATH='bigscience/mt0-large'
OUTPUT_DIR='output/pre-finetune'
BZ=4

python -m zalo_ai_math.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --train_dir 'data/pre-finetune/train' \
    --valid_dir 'data/pre-finetune/validation' \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 10 \
    --max_len 256 \
    --save_steps 200 \
    --eval_steps 1000 \
    --logging_steps 100