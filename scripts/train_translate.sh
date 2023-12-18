MODEL_NAME_OR_PATH='google/flan-t5-large'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    --config_file zalo_ai_math/accelerate_ds.yml -m zalo_ai_math.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \