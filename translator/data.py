
import logging

from datasets import Dataset, load_dataset, interleave_datasets
from translator.features.utils import multi_trans, a_2_b



logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, tokenizer, batch_size, data_args, seed) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_args = data_args
        self.seed = seed
        
    def __call__(self):
        dataset = {}

        # train data
        if self.data_args.dataset_name_train is not None:
            train_data = load_dataset(
                self.data_args.dataset_name_train,
                split = 'train',
                streaming = self.data_args.streaming,
                token = self.data_args.hf_key,
                # use_auth_token=self.data_args.use_auth_token,
            )
            if self.data_args.max_train_samples is not None:
                train_data = list(train_data.take(self.data_args.max_train_samples))
                train_data = Dataset.from_list(train_data)
                train_data = multi_trans(train_data, "en", "vi")
            else:
                en_2_vi = train_data.map(
                    a_2_b,
                    fn_kwargs={"language_a": "en", "language_b": "vi"},
                    batched=True,
                    remove_columns=['en', 'vi']
                )
                vi_2_en = train_data.map(
                    a_2_b,
                    fn_kwargs={"language_a": "vi", "language_b": "en"},
                    batched=True,
                    remove_columns=['en', 'vi']
                )
                en_2_vi = en_2_vi.shuffle(seed=self.seed, buffer_size=10_000)
                vi_2_en = vi_2_en.shuffle(seed=self.seed+1, buffer_size=10_000)
                train_data = interleave_datasets([en_2_vi, vi_2_en], seed=self.seed)
            dataset['train'] = self.process_fn(train_data)
        else:
            raise Exception(f'Not found `dataset_name_train` path.')
            
        # validation data
        if self.data_args.dataset_name_validation is not None:
            valid_data = load_dataset(
                self.data_args.dataset_name_validation,
                split = 'validation',
                token = self.data_args.hf_key,
            )
            if self.data_args.max_valid_samples is not None:
                valid_data = valid_data.select(range(self.data_args.max_valid_samples))
            valid_data = multi_trans(valid_data, "en", "vi")
            dataset['validation'] = self.process_fn(valid_data)
        else:
            raise Exception(f'Not found `dataset_name_validation` path.')
        
        return dataset
    
    def process_fn(self, dataset:Dataset) -> Dataset:
        """ Processing tokenizer 

        Args:
            datasets (Dataset): _description_

        Returns:
            Dataset tokenized
        """
        
        if self.data_args.streaming:
            dataset = dataset.map(
                lambda example : self.group_fn(example),
                remove_columns=['inputs', 'targets'],
            )
        else:
            dataset = dataset.map(
                lambda example : self.group_fn(example),
                num_proc=self.data_args.dataset_num_workers,
                remove_columns=['inputs', 'targets'],
            )
        
        return dataset
    
    def group_fn(self, example):
        # inputs
        model_inputs = self.tokenize_fn(example['inputs'], 
                                        length=self.data_args.max_len)
        # labels
        labels = self.tokenize_fn(example['targets'], 
                                          length=self.data_args.max_len, target=True)
        # print(labels)
        labels["input_ids"] = [
            # [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]

        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def tokenize_fn(self, x:str=None, length:int=None, padding=True, target=False):
        if target == False:
            return self.tokenizer(
                x,
                max_length=None if length is None else length,
                padding="max_length", truncation=True 
            )
        else:
            return self.tokenizer(
                text_target=x,
                max_length=None if length is None else length,
                padding="max_length", truncation=True 
            )