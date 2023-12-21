MODEL_NAME_OR_PATH="google/mt5-large"
# MODEL_NAME_OR_PATH="model/pt_full_model"
HF_TOKEN='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' 
DATASET_NAME_TRAIN ='presencesw/hash_20_256_v2'
DATASET_NAME_VALIDATION='presencesw/phomt_eval'
OUTPUT_DIR='outputs'
SAVE_DIR='model'
BZ=2
ALL_STEP=50_000_000
SAVE_EVAL_STEP=2_000_000


!WANDB_PROJECT=translator WANDB_API_KEY=138c38699b36fb0223ca0f94cde30c6d531895ca CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids all --num_processes 2 \
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
    --per_device_train_batch_size $BZ \
    --per_device_eval_batch_size $BZ \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 1\
    --max_len 256 \
    --save_steps $SAVE_EVAL_STEP \
    --eval_steps $SAVE_EVAL_STEP \
    --logging_steps 1000 \
    --save_total_limit 1 \
    --evaluation_strategy 'steps' \
    --att_blocks \
    --overwrite_output_dir \
    --load_best_model_at_end