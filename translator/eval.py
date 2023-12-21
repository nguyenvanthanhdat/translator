# import torch
# from peft import PeftModel, PeftConfig, LoraConfig
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
)
import transformers
# from adapters import AutoAdapterModel
from datasets import load_dataset
from argparse import ArgumentParser
import os, sys, logging
from .arguments import ModelArguments, DataTrainingArguments, LoraArguments
import evaluate

def get_output(examples, model, tokenizer, max_length, num_beams):
    prefix = examples['inputs'].strip()
    inputs = tokenizer.encode(prefix, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, 
                            #  max_new_tokens=max_length,
                             max_length=max_length,
                             num_beams=num_beams,
                             early_stopping=True)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    examples['translated'] = outputs
    return examples

def preprocess(examples, language_a = 'en', language_b = 'vi'):
    
    # change dataset to inputs, !!! not has targets 
    examples['inputs'] = [f'{language_a}: {sample}' for sample in examples[language_a]] \
        + [f'{language_a}: {sample}' for sample in examples[language_b]]
    examples['targets'] = [f'{language_b}: {sample}' for sample in examples[language_b]] \
        + [f'{language_a}: {sample}' for sample in examples[language_a]]
    return examples

def postprocess(examples):
    for exp in examples['translated']:
        if exp[:4] in ["vi: ", "en: "]:
            exp = exp[4:]
    return examples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_test_path', default='.', required=True)
    parser.add_argument('--split', default=None, required=True)
    parser.add_argument('--streaming', default=False, required=False)
    parser.add_argument('--hf_key', default=None, required=True)
    parser.add_argument('--model_name_or_path', default=None, required=True)
    parser.add_argument('--tokenizer_name_or_path', default=None, required=True)
    parser.add_argument('--num_proc', default=2, required=False, type=int)
    parser.add_argument('--max_length', default=256, required=False, type=int)
    parser.add_argument('--num_beams', default=5, required=False, type=int)
    args = parser.parse_args()

    path = os.getcwd()
    
    # load dataset to evaluate
    print("*"*20,"Load dataset","*"*20)
    if os.path.isdir(os.path.join(path, args.data_test_path)):
        dataset = load_dataset(
            os.path.join(path, args.data_test_path),
            split = args.split,
            token = args.hf_key,
            num_proc=args.num_proc,
        )
    else:
        dataset = load_dataset(
            args.data_test_path,
            split = args.split,
            token = args.hf_key,
            num_proc=args.num_proc,
        )
    print("*"*20,"Load dataset Done","*"*20)
    
    # preprocess dataset ['en', 'vi'] -> ['inputs']
    print("*"*20,"Preprocess data","*"*20)
    dataset = dataset.map(preprocess, remove_columns=['en', 'vi'], batched=True)

    # Load model
    if os.path.isdir(os.path.join(path, args.model_name_or_path)):
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, args.model_name_or_path)).to('cuda')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to('cuda')

    # Load tokenizer
    if os.path.isdir(os.path.join(path, args.tokenizer_name_or_path)):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, args.model_name_or_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # model = PeftModel.from_pretrained(model, args.adapters_path)
    print("*"*20,"Translate ...","*"*20)
    dataset = dataset.map(get_output,
                fn_kwargs={"tokenizer": tokenizer, "model": model, 
                           "max_length": args.max_length, "num_beams": args.num_beams},
                remove_columns=['inputs'])
    
    print("*"*20,"Postprocess data","*"*20)
    dataset = dataset.map(postprocess, batched=True)

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=dataset['translated'], references=dataset['targets'])
    print(results)
    print("*"*20,"ALL DONE","*"*20)