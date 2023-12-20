import logging
import datasets
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


EXPL_PROMPT = """Please act as an expert in multi-lingual understanding $SOURCE_LANG.
Request: $REQUEST.
Let's understand the task in $TARGET_LANG step by step!"""
PRED_PROMPT = """After understanding, you should act as an expert in $TARGET_LANG. 
Let’s resolve the task you understand above step-by-step!
Finally, you should format your answer as 'Answer: [num]'")"""
PRED_LABEL = """Sure! Let's solve the task step-by-step:\n $TARGET_COT. Answer: $SOURCE_LABEL"""

def is_filter_samples(input_str, filter_length=256) -> bool:
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/zalo_ai_math/checkpoint-5000-mt0-large")
    input = tokenizer.tokenize(input_str) # token_lists
    if len(input) > filter_length:
        return True
    return False

def a_2_b(examples, language_a, language_b):
    examples['inputs'] = [f'{language_a}: {sample}' for sample in examples[language_a]]
    examples['targets'] = [f'{language_b}: {sample}' for sample in examples[language_b]]
    return examples

def multi_trans(dataset, language_a, language_b):
    dataset_a_2_b = dataset.map(a_2_b, fn_kwargs={"language_a": language_a, "language_b": language_b}
                                , batched=True, remove_columns=[language_a, language_b])
    dataset_b_2_a = dataset.map(a_2_b, fn_kwargs={"language_a": language_b, "language_b": language_a}
                                , batched=True, remove_columns=[language_a, language_b])
    new_dataset = datasets.concatenate_datasets([dataset_a_2_b, dataset_b_2_a])
    return new_dataset

def multi_trans_steaming(examples, language_a, language_b):
    examples['inputs'] = [f'{language_a}: {sample}' for sample in examples[language_a]] \
        + [f'{language_a}: {sample}' for sample in examples[language_b]]
    examples['targets'] = [f'{language_b}: {sample}' for sample in examples[language_b]] \
        + [f'{language_a}: {sample}' for sample in examples[language_a]]
    return examples