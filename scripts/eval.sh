MODEL_NAME_OR_PATH="google/mt5-large"
# MODEL_NAME_OR_PATH="model/pt_full_model"
TOKEN_NAME='google/mt5-large'
HF_TOKEN='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' 
DATASET_TEST_PATH='presencesw/hash_20_256_v2'
SPLIT='test'
OUTPUT_DIR='outputs'
SAVE_DIR='model'
BZ=2
ALL_STEP=50_000_000
SAVE_EVAL_STEP=2_000_000


CUDA_VISIBLE_DEVICES=0 python -m translator.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name_or_path $TOKEN_NAME \
    --hf_key $HF_TOKEN \
    --data_test_path $DATA_TEST_PATH \
    --split $SPLIT \
    --streaming \
    --max_len 256 \
    --num_beams 256 \
    --hf_key $HF_TOKEN \