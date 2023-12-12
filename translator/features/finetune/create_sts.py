import os
import pandas as pd
from datasets import load_dataset, Dataset
from argparse import ArgumentParser

from .utils import *



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--craw_finetune_path', default='data/craw/finetune', required=False)
    parser.add_argument('--output_finetune_path', default='data/interim/finetune', required=False)
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

    # # GSMK8K
    # gsm8k_en_path = os.path.join(args.craw_finetune_path, 'GSM8K.en.json')
    # gsm8k_en = load_dataset('json', data_files=gsm8k_en_path, split='train')
    # gsm8k_vi_path = os.path.join(args.craw_finetune_path, 'GSM8K.vi.json')
    # gsm8k_vi = load_dataset('json', data_files=gsm8k_vi_path, split='train')
    
    # gsm8k_succses, gsm8k_error = [], []
    # for eng, vie in zip(gsm8k_en, gsm8k_vi):
    #     result = gsm8k_fn('Vietnamese', 'English', vie, eng)
    #     if len(result) < 3:
    #         gsm8k_error.append(result)
    #     else:
    #         gsm8k_succses.append(result)
    
    # gsm8k_output = os.path.join(args.output_finetune_path, 'gsm8k.json')
    # Dataset.from_pandas(pd.DataFrame(data=gsm8k_succses)).to_json(gsm8k_output, force_ascii=False)
    # print("\nReport GSM8K: \n\t\tSuccess samples:", len(gsm8k_succses), "\t Error samples:", len(gsm8k_error))
    
    # # AQUA
    # aqua_en_path = os.path.join(args.craw_finetune_path, 'AQUA.en.json')
    # aqua_en = load_dataset('json', data_files=aqua_en_path, split='train')
    # aqua_vi_path = os.path.join(args.craw_finetune_path, 'AQUA.vi.json')
    # aqua_vi = load_dataset('json', data_files=aqua_vi_path, split='train')
    
    # aqua_success, aqua_error = [], []
    # for eng, vie in zip(aqua_en, aqua_vi):
    #     result = aqua_fn('Vietnamese', 'English', vie, eng)
    #     if len(result) < 3:
    #         aqua_error.append(result)
    #     else:
    #         aqua_success.append(result)
    # aqua_output = os.path.join(args.output_finetune_path, 'aqua.json')
    # Dataset.from_pandas(pd.DataFrame(data=aqua_success)).to_json(aqua_output, force_ascii=False)
    # print("\nReport AQUA: \n\t\tSuccess samples:", len(aqua_success), "\t Error samples:", len(aqua_error))
    
    # # AQUA_ENHANCE
    # aqua_enhance_path = os.path.join(args.craw_finetune_path, 'AQUA.enhance.json')
    # aqua_enhance = load_dataset('json', data_files=aqua_enhance_path, split='train')
    
    # aqua_enhance_success = []
    # for row in aqua_enhance:
    #     aqua_enhance_success.append(aqua_enhance_fn('English', row))
    
    # aqua_enhance_output = os.path.join(args.output_finetune_path, 'aqua_enhance.json')
    # Dataset.from_pandas(pd.DataFrame(data=aqua_enhance_success)).to_json(aqua_enhance_output, force_ascii=False)
    # print("\nReport AQUA_ENHANCE: \n\t\tSuccess samples: ", len(aqua_enhance_success))
    
    # # ZALO
    # zalo_path = os.path.join(args.craw_finetune_path,'zalo_with_expl.json')
    # zalo = load_dataset('json', data_files=zalo_path, split='train')
    
    # zalo_success = []
    # for row in zalo:
    #     zalo_success.append(zalo_fn('Vietnamese', row))
    # zalo_output = os.path.join(args.output_finetune_path, 'zalo.json')
    # Dataset.from_pandas(pd.DataFrame(data=zalo_success)).to_json(zalo_output, force_ascii=False)
    # print("\nReport ZALO: \n\t\tSuccess samples", len(zalo_success))

    # # merge all
    # print("\n", "="*20, " Merge all dataset ", "="*20, "\n")
    # dataset = load_dataset('json', data_files=os.path.join(args.output_finetune_path, '*.json'), split='train')
    # print("\nTotal sample: ", dataset)
    
    # # Create train and validation field
    # for _ in range(100):
    #     dataset = dataset.shuffle(42)
    # dataset = dataset.train_test_split(test_size=0.05)
    # dataset['train'].to_json(os.path.join(args.output_finetune_path, 'merge_train.json'), force_ascii=False)
    # dataset['test'].to_json(os.path.join(args.output_finetune_path, 'merge_validation.json'), force_ascii=False)
    # print("\n", "="*20, " Create success ", "="*20, "\n")