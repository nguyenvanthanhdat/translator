MODEL_NAME_OR_PATH="google/mt5-small"
# MODEL_NAME_OR_PATH="model/pt_full_model"
TOKEN_NAME='google/mt5-small'
HF_TOKEN='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' 
DATA_TEST_PATH='presencesw/phomt_eval'
SPLIT='test'
OUTPUT_DIR='outputs'
SAVE_DIR='model'
BZ=2
ALL_STEP=50_000_000
SAVE_EVAL_STEP=2_000_000


CUDA_VISIBLE_DEVICES=0 python -m translator.eval \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name_or_path $TOKEN_NAME \
    --hf_key $HF_TOKEN \
    --data_test_path $DATA_TEST_PATH \
    --split $SPLIT \
    --num_proc 1 \
    --max_len 256 \
    --num_beams 5 \
    --hf_key $HF_TOKEN \