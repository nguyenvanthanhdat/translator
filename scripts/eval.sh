MODEL_TEST_PATH = .
MODEL_NAME_OR_PATH = bigscience/mt0-large
ADAPTERS_PATH = D:\Workspace\zalo_ai_math\output\\finetune\checkpoint-10

python -m zalo_ai_math.eval \
    --data_test_path $MODEL_TEST_PATH \
    --model_or_path $MODEL_NAME_OR_PATH \
    --adapters_path $ADAPTERS_PATH