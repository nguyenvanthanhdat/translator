# Dataset

To down dataset test (1000 samples), please run `sh scripts/down_data.sh`

# Prepocess Data

The step to change dataset from 2 columns ['en', 'vi'] to ['inputs', 'targets'] 2 direction vi -> en and en -> vi

```python
python -m translator.features.finetune.create_sts
```

# Fine-tune Model

Run the scripts below model

```shell
MODEL_NAME_OR_PATH="google/mt5-large"
OUTPUT_DIR='output'
SAVE_DIR='model'
BZ=4


!WANDB_PROJECT=translator WANDB_API_KEY=138c38699b36fb0223ca0f94cde30c6d531895ca accelerate launch --gpu_ids all --num_processes 1 \
    --config_file translator/accelerate_ds.yml -m translator.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --train_dir 'data/finetune/dataset_1000' \
    --do_eval \
    --valid_dir 'data/finetune/dataset_1000' \
    --quantize \
    --use_lora \
    --per_device_train_batch_size $BZ \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 1\
    --max_len 256 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --evaluation_strategy 'steps' \
    --step_by_step \
    --target_modules \
    --att_blocks \
    --overwrite_output_dir \
    --load_best_model_at_end
```
