MODEL_NAME_OR_PATH="google/mt5-large"
# MODEL_NAME_OR_PATH="model/pt_full_model"
TOKEN_NAME='google/mt5-large'
HF_TOKEN='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' 
# DATA_TEST_PATH='presencesw/phomt_eval'
DATA_TEST_PATH='presencesw/hash_test'
SPLIT='test'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes=2 -m translator.eval \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name_or_path $TOKEN_NAME \
    --hf_key $HF_TOKEN \
    --data_test_path $DATA_TEST_PATH \
    --split $SPLIT \
    --num_proc 2 \
    --batch_size 1 \
    --max_len 512 \
    --num_beams 3,4,5 \
    --hf_key $HF_TOKEN \
    --use_lora True 