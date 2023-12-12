import logging
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


def gsm8k_fn(source, target, example_src, example_tgt):
    # explanation inputs
    expl_prompt = EXPL_PROMPT.replace('$SOURCE_LANG', source.strip())\
                            .replace('$REQUEST', example_src['question'].strip())\
                            .replace('$TARGET_LANG', target.strip())
    cot_srcs = example_src['cot'].split('.')
    cot_tgts = example_tgt['cot'].split('.')
    if len(cot_srcs) != len(cot_tgts):
        return [example_src, example_tgt]

    # examplation labels
    tmp = []
    for idx, (cot_src, cot_tgts) in enumerate(zip(cot_srcs, cot_tgts)):
        if len(cot_src) > 1 and len(cot_tgts) > 1:
            tmp.append(f'{idx+1}. "{cot_src.strip()}" means "{cot_tgts.strip()}"')
    expl_label = '\n'.join(tmp)

    # question - answer
    pred_prompt = PRED_PROMPT.replace('$TARGET_LANG', target.strip())
    pred_label = PRED_LABEL.replace('$TARGET_COT', example_tgt['cot'].strip())\
                            .replace('$SOURCE_LABEL', str(example_src['answer']).strip())
    
    if is_filter_samples(expl_prompt) or is_filter_samples(pred_prompt) or is_filter_samples(pred_label):
        return {}

    return {
        'input_expl' : expl_prompt.strip(),
        'label_expl' : expl_label.strip(),
        'input_pred' : pred_prompt.strip(),
        'label_pred' : pred_label.replace("..", ".").strip()
    }
    
def aqua_fn(source, target, example_src, example_tgt):
    # explanation inputs
    expl_prompt = EXPL_PROMPT.replace('$SOURCE_LANG', source.strip())\
                            .replace('$REQUEST', example_src['question'].strip())\
                            .replace('$TARGET_LANG', target.strip())

    cot_srcs = example_src['cot'].split('\n')
    cot_tgts = example_tgt['cot'].split('\n')

    if len(cot_srcs) != len(cot_tgts):
        return [example_src, example_tgt]
    # examplation labels
    tmp = []
    for idx, (cot_src, cot_tgts) in enumerate(zip(cot_srcs, cot_tgts)):
        if len(cot_src) > 1 and len(cot_tgts) > 1:
            tmp.append(f'{idx+1}. "{cot_src.strip()}" means "{cot_tgts.strip()}"')
    expl_label = '\n'.join(tmp)

    # question - answer
    pred_prompt = PRED_PROMPT.replace('$TARGET_LANG', target.strip())
    pred_label = PRED_LABEL.replace('$TARGET_COT', example_tgt['cot'].strip())\
                            .replace('$SOURCE_LABEL', str(example_src['answer']).strip())

    if is_filter_samples(expl_prompt) or is_filter_samples(pred_prompt) or is_filter_samples(pred_label):
        return {}

    return {
        'input_expl' : expl_prompt.replace('Options:\n', '[')\
                                    .replace("\nLet's ", "]\nLet's ") \
                                    .replace("\n( ", ", "),
        'label_expl' : expl_label.strip(),
        'input_pred' : pred_prompt.strip(),
        'label_pred' : pred_label.replace('..', '.').strip()
    }
    
def aqua_enhance_fn(source, example):
    # explanation inputs
    expl_prompt = EXPL_PROMPT.replace('$SOURCE_LANG', source.strip())\
                            .replace('$REQUEST', example['question'].strip() + ' ' + str(example['options']))\
                            .replace('$TARGET_LANG', source.strip())
    cot = []
    for idx, item in enumerate(example['rationale'].split('\n')):
        if len(item) > 1:
            cot.append(f'{idx+1}. {item.strip()}')
    # examplation labels
    expl_label = '\n'.join(cot)

    # question - answer
    pred_prompt = PRED_PROMPT.replace('$TARGET_LANG', source.strip())
    pred_label = PRED_LABEL.replace('$TARGET_COT', example['rationale'].strip())\
                            .replace('$SOURCE_LABEL', str(example['answer']).strip())

    if is_filter_samples(expl_prompt) or is_filter_samples(pred_prompt) or is_filter_samples(pred_label):
        return {}

    return {
        'input_expl' : expl_prompt.strip(),
        'label_expl' : expl_label.strip(),
        'input_pred' : pred_prompt.strip(),
        'label_pred' : pred_label.replace('..', '.').strip()
    }
    
def zalo_fn(source, example):
    # explanation inputs
    expl_prompt = EXPL_PROMPT.replace('$SOURCE_LANG', source.strip())\
                            .replace('$REQUEST', example['question'].strip() + ' ' + str(example['choices']))\
                            .replace('$TARGET_LANG', source.strip())
    cot= []
    tmp = example['explanation'].split('.')
    if len(tmp) == 1:
        tmp = example['explanation'].split('\n')
    # examplation labels
    for idx, item in enumerate(tmp):
        if len(item) > 1 and not item.startswith(' Đáp số'):
            cot.append(f'{idx+1}. {item.strip()}')
    expl_label = '\n'.join(cot)

    # question - answer 
    pred_prompt = PRED_PROMPT.replace('$TARGET_LANG', source.strip())
    pred_label = PRED_LABEL.replace('$TARGET_COT', example['explanation'].strip())\
                            .replace('$SOURCE_LABEL', str(example['answer']).strip())

    if is_filter_samples(expl_prompt) or is_filter_samples(pred_prompt) or is_filter_samples(pred_label):
        return {}

    return {
        'input_expl' : expl_prompt.strip(),
        'label_expl' : expl_label.strip(),
        'input_pred' : pred_prompt.strip(),
        'label_pred' : pred_label.replace('..', '.').strip()
    }