MODEL_NAME_OR_PATH="google/mt5-large"
DATASET_NAME_TRAIN ='presencesw/hash_20_256_v2'
DATASET_NAME_VALIDATION='presencesw/phomt_eval'
OUTPUT_DIR='outputs'
SAVE_DIR='model'
BZ=2


!WANDB_PROJECT=translator WANDB_API_KEY=138c38699b36fb0223ca0f94cde30c6d531895ca accelerate launch --gpu_ids all --num_processes 2 \
    --config_file translator/accelerate_ds.yml -m translator.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --dataset_name_train $DATASET_NAME_TRAIN \
    --do_eval \
    --dataset_name_validation $DATASET_NAME_VALIDATION \
    --max_train_samples 1000 \
    --streaming \
    --use_lora \
    --per_device_train_batch_size $BZ \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 1\
    --max_len 256 \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 100 \
    --evaluation_strategy 'steps' \
    --step_by_step \
    --target_modules \
    --att_blocks \
    --overwrite_output_dir \
    --load_best_model_at_end