MODEL_NAME_OR_PATH='bigscience/mt0-large'
OUTPUT_PRE_FINETUNE='output/pre-finetune'
OUTPUT_FINETUNE='output/pre-finetune'
BZ=4

python -m zalo_ai_math.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_PRE_FINETUNE \
    --train_dir 'data/pre-finetune/train' \
    --valid_dir 'data/pre-finetune/validation' \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 10 \
    --max_len 256 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --logging_steps 100


python -m zalo_ai_math.training \
    --model_name_or_path $OUTPUT_PRE_FINETUNE \
    --output_dir $OUTPUT_FINETUNE \
    --do_train \
    --train_dir 'data/finetune/train' \
    --do_eval \
    --valid_dir 'data/finetune/validation' \
    --quantize \
    --use_lora \
    --per_device_train_batch_size $BZ \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 4 \
    --max_len 256 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --evaluation_strategy 'steps' \
    --step_by_step \
    --overwrite_output_dir \
    --load_best_model_at_end