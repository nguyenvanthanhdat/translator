import os
import glob
import logging

from datasets import Dataset, load_dataset, load_from_disk
from transformers import DataCollatorForSeq2Seq


logger = logging.getLogger(__name__)



class Processor:
    def __init__(self, tokenizer, batch_size, data_args) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_args = data_args
        
    def __call__(self):
        dataset = {}
        # train set
        if self.data_args.train_dir is not None:
            train_data = self.load_data(self.data_args.train_dir, 'train')
            
            if self.data_args.max_train_samples is not None:
                train_data = train_data.select(range(self.data_args.max_train_samples))
            
            dataset['train'] = self.process_fn(train_data)
            
        # validation set
        if self.data_args.valid_dir is not None:
            valid_data = self.load_data(self.data_args.valid_dir, 'validation')
        
            if self.data_args.max_valid_samples is not None:
                valid_data = valid_data.select(range(self.data_args.max_valid_samples))
                
            dataset['validation'] = self.process_fn(valid_data)
        return dataset
    
    def load_data(self, data_path:str=None, key:str='train') -> Dataset:
        """ Load datasets function 

        Args:
            data_path (str, optional): folder contain list of input files name. Defaults to None.
            key (str, optional): help dataloader know is train file or test file. 
                                Input file can be train/validation/test. Defaults to 'train'.

        Raises:
            Exception: _description_

        Returns:
            Datasets
        """
        if not os.path.exists(data_path):
            raise ValueError(f'Not found {data_path} path.')
        
        # files = glob.glob(os.path.join(data_path, '*'))
        # extention = files[0].split('.')[-1]
        try:
            # data_file = f"{data_path}/*.{extention}"
            data_file = data_path
            print(data_path)
            if self.data_args.streaming:
                dataset = load_from_disk(
                    dataset_path=data_file
                )[key]
            else:
                dataset = load_from_disk(
                    dataset_path=data_file
                )[key]
            return dataset

        except:
            logger.info(f'Error loading dataset {data_path}')
            print(f'Error loading dataset {data_path}')
    
    def process_fn(self, datasets:Dataset) -> Dataset:
        """ Processing tokenizer 

        Args:
            datasets (Dataset): _description_

        Returns:
            Dataset tokenized
        """
        
        if self.data_args.streaming:
            datasets = datasets.map(
                lambda example : self.group_fn(example),
                remove_columns=['inputs', 'targets'],
            )
        else:
            datasets = datasets.map(
                lambda example : self.group_fn(example),
                num_proc=self.data_args.dataset_num_workers,
                remove_columns=['inputs', 'targets'],
            )
        
        return datasets
    
    def group_fn(self, example):
        # inputs
        model_inputs = self.tokenize_fn(example['inputs'], 
                                        length=self.data_args.max_len)
        # labels
        labels = self.tokenize_fn(example['targets'], 
                                          length=self.data_args.max_len)
        # print(labels)
        labels["input_ids"] = [
            # [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]

        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def tokenize_fn(self, x:str=None, length:int=None, padding=True):
        return self.tokenizer(
          x,
          max_length=None if length is None else length,
          padding=padding, truncation=True 
        )
    
        
# class SBSProcessor(Processor):
#     def process_fn(self, datasets:Dataset) -> Dataset:
#         """ Processing tokenizer 

#         Args:
#             datasets (Dataset): _description_

#         Returns:
#             Dataset tokenized
#         """
        
#         if self.data_args.streaming:
#             datasets = datasets.map(
#                 lambda example : self.group_fn(example),
#                 # lambda example : self.group_fn(example),
#                 # remove_columns=['input_pred', 'label_pred', 'input_expl', 'label_expl'],
#             )
#         else:
#             datasets = datasets.map(
#                 lambda example : self.group_fn(example),
#                 # lambda example : self.group_fn(example),
#                 num_proc=self.data_args.dataset_num_workers,
#                 # remove_columns=['input_pred', 'label_pred', 'input_expl', 'label_expl'],
#             )
        
#         return datasets
    
#     def group_fn(self, example):
#         # question
#         model_inputs = self.tokenize_fn(example['input_pred'],length=self.data_args.max_len)
#         # answer
#         labels = self.tokenize_fn(example['label_pred'],length=self.data_args.max_len)
#         labels["input_ids"] = [
#             (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]
#         ]
#         model_inputs["labels"] = labels["input_ids"]

#         # CoT
#         cot_inputs = self.tokenize_fn(example['input_expl'],length=self.data_args.max_len)
#         # CoT label
#         cot_labels = self.tokenize_fn(example['label_expl'], length=self.data_args.max_len)
#         cot_labels["input_ids"] = [
#             (l if l != self.tokenizer.pad_token_id else -100) for l in cot_labels["input_ids"]
#         ]
#         cot_inputs['labels'] = cot_labels["input_ids"]
 
#         return {
#             'pred' : model_inputs,
#             'expl' : cot_inputs
#         }
    

# class SBSDataCollator(DataCollatorForSeq2Seq):
#     def __call__(self, features, return_tensors=None):
    
#         pred_features = [x['pred'] for x in features]
#         expl_features = [x['expl'] for x in features]
        
#         pred_inputs = super().__call__(pred_features, return_tensors)
#         expl_inputs = super().__call__(expl_features, return_tensors)

#         return {
#             'pred' : pred_inputs,
#             'expl' : expl_inputs
#         }