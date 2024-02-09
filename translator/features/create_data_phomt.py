import os
from datasets import load_dataset, Dataset
from argparse import ArgumentParser

from .utils import *



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--craw_finetune_path', default='data/craw', required=False)
    parser.add_argument('--output_finetune_path', default='data/finetune', required=False)
    args = parser.parse_args()
    
    # dataset PhoMT
    phomt_path = os.path.join(args.craw_finetune_path, "dataset_1000")
    phomt_save_path = os.path.join(args.output_finetune_path, "dataset_1000")
    phomt = Dataset.load_from_disk(phomt_path)
    phomt = multi_trans(phomt, "en", "vi")
    phomt = phomt.train_test_split(test_size=0.1, shuffle=False)
    phomt = datasets.DatasetDict({
        'train': phomt['train'],
        'validation': phomt['test']})
    print(phomt['train'][0])
    print(phomt['train'][1])
    phomt.save_to_disk(phomt_save_path)