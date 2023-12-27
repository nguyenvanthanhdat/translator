from datasets import (
    load_from_disk,
    load_dataset,
    Features,
    Value,
    concatenate_datasets,
)
from transformers import (
    AutoTokenizer,
)
import gdown

def add_column_cs(examples, indx):
    examples['vi'] = pubmed_vi[indx]['text']
    return examples

def get_token(vi_texts):
    input_ids = tokenizer(vi_texts)
    return input_ids

def tokenized(examples):
  tokenized_vi = get_token(examples['vi'])['input_ids']
  tokenized_en = get_token(examples['en'])['input_ids']
  examples['len_vi'] = [len(i) for i in tokenized_vi]
  examples['len_en'] = [len(i) for i in tokenized_en]
  return examples 

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


dataset_phomt = load_from_disk("PhoMT_detokenization")
print('dataset_phomt')
print(dataset_phomt)

data_path = 'datasets--justinphan3110--vi_pubmed/'
context_feat = Features({'text': Value(dtype='string', id=None)})
pubmed_en = dataset = load_dataset("parquet", 
    data_files=data_path+"en-*.parquet", 
    features=context_feat,
    num_proc=4,
    split='train',
)

pubmed_vi = load_dataset("parquet", 
    data_files=data_path+"vi-*.parquet", 
    features=context_feat,
    num_proc=4,
    split='train',
)

pubmed_en = pubmed_en.rename_column("text", "en")

pubmed_envi = pubmed_en.map(add_column_cs, with_indices=True, batched=True)

dataset_envi = concatenate_datasets([dataset_phomt, pubmed_envi])

dataset_envi_tokenized = dataset_envi.map(tokenized, batched=True)

dataset_en_better_than_vi = dataset_envi_tokenized.filter(lambda example: example["len_en"] > example["len_vi"])

print(f"The number sample token_en > token_vi is {len(dataset_en_better_than_vi)}")

dataset_envi_tokenized_20_256 = dataset_envi_tokenized.filter(lambda example: example["len_vi"] >= 20 and example["len_vi"] <= 256)

print(f"The number sample in range 20 - 256 is {len(dataset_envi_tokenized_20_256)}")

dataset_envi_tokenized_256_512 = dataset_envi_tokenized.filter(lambda example: example["len_vi"] > 256 and example["len_vi"] <= 512)

print(f"The number sample in range 257 - 512 is {len(dataset_envi_tokenized_256_512)}")

HF_TOKEN='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'

dataset_envi_v3 = concatenate_datasets([dataset_envi_tokenized_20_256, dataset_envi_tokenized_256_512]).remove_columns(["len_vi", "len_en"])

for _ in range(20):
    dataset_envi_v3 = dataset_envi_v3.shuffle(seed=42)

dataset_envi_v3.push_to_hub(
    "hash_20_256_v3",
    token = HF_TOKEN,
    private=True,
)