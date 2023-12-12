import os
from datasets import load_dataset, concatenate_datasets

def main():
    path = os.getcwd()

    en_train = load_dataset("json", data_files=os.path.join(path, "data/interim/pre-finetune/en_train.json"))['train']
    en_test = load_dataset("json", data_files=os.path.join(path, "data/interim/pre-finetune/en_test.json"))['train']
    vi_train = load_dataset("json", data_files=os.path.join(path, "data/interim/pre-finetune/vi_train.json"))['train']
    vi_test = load_dataset("json", data_files=os.path.join(path, "data/interim/pre-finetune/vi_test.json"))['train']
    train = concatenate_datasets([en_train, vi_train])
    test = concatenate_datasets([en_test, vi_test])
    for i in range(20):
        train = train.shuffle(seed=42)
        test = test.shuffle(seed=42)
    train.to_json(os.path.join(path, "data/interim/pre-finetune/train.json"))
    test.to_json(os.path.join(path, "data/interim/pre-finetune/test.json"))

if __name__ == "__main__":
    main()