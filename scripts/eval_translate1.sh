DATASET_NAME='presencesw/dataset1_translated'
# SPLIT='test'


CUDA_VISIBLE_DEVICES=0 python -m translator.eval_translate_llm \
    --dataset_name $DATASET_NAME