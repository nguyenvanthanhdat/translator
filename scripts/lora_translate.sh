MODEL_NAME_OR_PATH="google/mt5-large"
HF_TOKEN='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' 
DATASET_NAME_TRAIN=presencesw/hash_20_256_v2
DATASET_NAME_VALIDATION=presencesw/phomt_eval
OUTPUT_DIR='outputs'
SAVE_DIR='model'
BZ=3
ALL_STEP=100000
SAVE_EVAL_STEP=5000
GRA_ACC=14
LORA_R=16
LORA_ALPHA=16


WANDB_PROJECT=translator WANDB_API_KEY=138c38699b36fb0223ca0f94cde30c6d531895ca CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --gpu_ids 0,1,2,3 --num_processes 4 \
    --config_file translator/accelerate_ds.yml -m translator.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --hf_key $HF_TOKEN \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --dataset_name_train $DATASET_NAME_TRAIN \
    --do_eval \
    --dataset_name_validation $DATASET_NAME_VALIDATION \
    --max_steps $ALL_STEP \
    --streaming \
    --use_lora \
    --linear_layer \
    --att_blocks \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --per_device_train_batch_size $BZ \
    --per_device_eval_batch_size $BZ \
    --num_train_epochs 4 \
    --gradient_accumulation_steps $GRA_ACC \
    --max_len 256 \
    --save_steps $SAVE_EVAL_STEP \
    --eval_steps $SAVE_EVAL_STEP \
    --logging_steps 100 \
    --save_total_limit 3 \
    --evaluation_strategy 'steps' \
    --learning_rate 3e-5 \
    --report_to 'wandb' \
    --overwrite_output_dir \
    --load_best_model_at_end