# HF_TOKEN='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' 
# DATA_TEST_PATH='presencesw/phomt_eval'
DATASET_NAME='presencesw/dataset4'
# SPLIT='test'


CUDA_VISIBLE_DEVICES=1 python -m translator.translate \
    --dataset_name $DATASET_NAME \
    --batch_size 15