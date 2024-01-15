# HF_TOKEN='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' 
# DATA_TEST_PATH='presencesw/phomt_eval'
DATASET_NAME='presencesw/dataset1'
# SPLIT='test'


CUDA_VISIBLE_DEVICES=0 python -m translator.translate \
    --dataset_name $DATASET_NAME \
    --batch_size 20