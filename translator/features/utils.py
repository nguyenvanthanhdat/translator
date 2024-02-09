import logging
import datasets



logger = logging.getLogger(__name__)


def a_2_b(examples, language_a, language_b):
    examples['inputs'] = [f'{language_a}: {sample} <EOS>' for sample in examples[language_a]]
    examples['targets'] = [f'{language_b}: {sample} <EOS>' for sample in examples[language_b]]
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