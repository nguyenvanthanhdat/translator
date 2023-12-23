# import torch
# from peft import PeftModel, PeftConfig, LoraConfig
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    T5Tokenizer
)
import transformers
# from adapters import AutoAdapterModel
from datasets import load_dataset
from argparse import ArgumentParser
import os, sys, logging
from .arguments import ModelArguments, DataTrainingArguments, LoraArguments
import evaluate
from accelerate import PartialState

def get_output(examples, model, tokenizer, max_length, num_beams):
    prefix = [exp.strip() for exp in examples['input']]
    inputs = tokenizer(
        prefix, return_tensors="pt",
        padding=True
    ).to("cuda")
    outputs = model.generate(**inputs, 
                            #  max_new_tokens=max_length,
                             max_length=max_length,
                             num_beams=num_beams,
                             early_stopping=True)
    outputs = [output[0] for output in outputs]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    examples['predict'] = outputs
    return examples

def preprocess(examples, language_a, language_b):
    
    # change dataset to inputs, !!! not has targets 
    examples['input'] = [f'{language_a}: {sample}' for sample in examples[language_a]]
    examples['target'] = [f'{language_b}: {sample}' for sample in examples[language_b]] 
    return examples

def postprocess(examples):
    for exp in examples['predict']:
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
    parser.add_argument('--batch_size', default=100, required=False, type=int)
    parser.add_argument('--max_length', default=256, required=False, type=int)
    parser.add_argument('--num_beams', default=5, required=False)
    args = parser.parse_args()

    path = os.getcwd()
    eval_path = os.path.join(os.getcwd(), "/eval")
    
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
    # print("*"*20,"Preprocess data","*"*20)
    # dataset = dataset.map(preprocess, remove_columns=['en', 'vi'], batched=True)

    # Load model
    

    # Load tokenizer
    if os.path.isdir(os.path.join(path, args.tokenizer_name_or_path)):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, args.model_name_or_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # model = PeftModel.from_pretrained(model, args.adapters_path)
    # print("*"*20,"Translate ...","*"*20)
    # num_beams = args.num_beams.split(",")
    # print("*"*20,f"Preprocess data","*"*20)
    # dataset_envi = dataset.map(
    #     preprocess, remove_columns=["en", "vi"], batched=True,
    #     fn_kwargs={"language_a":"en","language_b":"vi"}
    # )

    # dataset_vien = dataset.map(
    #     preprocess, remove_columns=["en", "vi"], batched=True,
    #     fn_kwargs={"language_a":"vi","language_b":"en"}
    # )
    distributed_state = PartialState()
    with distributed_state.split_between_processes(["en->vi", "vi->en"]) as distribute:
        distribute = distribute[0].split("->")
        language_a = distribute[0]
        language_b = distribute[1]

        if os.path.isdir(os.path.join(path, args.model_name_or_path)):
            model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, args.model_name_or_path)).to('cuda')
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to('cuda')
        for num_beam in args.num_beams.split(","): # [3, 4, 5]
            print("*"*20,f"Translate with num_bema = {num_beam}, {language_a} -> {language_b} ...","*"*20)
            print(dataset)
            print("*"*20,f"Preprocess data","*"*20)
            proprocess_dataset = dataset.map(
                preprocess, remove_columns=["en", "vi"], batched=True,
                fn_kwargs={"language_a": language_a,"language_b":language_b}
            )
            dataset_translated = proprocess_dataset.map(get_output,
                        fn_kwargs={"tokenizer": tokenizer, "model": model, 
                                "max_length": args.max_length, "num_beams": int(num_beam)},
                        batched=True,
                        batch_size=args.batch_size,
                        remove_columns=['input'])
            print("*"*20,"Postprocess data","*"*20)
            # print(dataset_envi)
            dataset_translated = dataset_translated.map(postprocess, batched=True)
            dataset_translated.to_json(os.path.join(eval_path,f"{language_a}{language_b}-beam{num_beam}.txt"))

            

        # if distribute == 0:
        #     language_a = "en"
        #     language_b = "vi"
        #     print("*"*20,f"Translate with num_bema = {num_beam}, {language_a} -> {language_b} ...","*"*20)
        #     print(dataset_envi)
        #     dataset_envi = dataset_envi.map(get_output,
        #                 fn_kwargs={"tokenizer": tokenizer, "model": model, 
        #                         "max_length": args.max_length, "num_beams": int(num_beam)},
        #                 batched=True,
        #                 batch_size=args.batch_size,
        #                 remove_columns=['input'])
        #     print(dataset_envi)
        # if distribute == 1:
        #     language_a = "vi"
        #     language_b = "en"
        #     print("*"*20,f"Translate with num_bema = {num_beam}, {language_a} -> {language_b} ...","*"*20)
        #     dataset_vien = dataset_vien.map(get_output,
        #                 fn_kwargs={"tokenizer": tokenizer, "model": model, 
        #                         "max_length": args.max_length, "num_beams": int(num_beam)},
        #                 batched=True,
        #                 batch_size=args.batch_size,
        #                 remove_columns=['input'])
        # try:
        #     print("*"*20,"Postprocess data","*"*20)
        #     # print(dataset_envi)
        #     dataset_envi = dataset_envi.map(postprocess, batched=True)
        #     print(os.path.join(eval_path,f"{language_a}{language_b}-beam{num_beam}.txt"))
        #     # dataset_envi.to_json(os.path.join(eval_path,f"{language_a}{language_b}-beam{num_beam}.txt")) 
        #     dataset_envi.to_json("a.json") 
        # except:
        #     print("*"*20,"Postprocess data","*"*20)
        #     dataset_vien = dataset_vien.map(postprocess, batched=True)
        #     dataset_vien.to_json(os.path.join(eval_path,f"{language_a}{language_b}-beam{num_beam}.txt"))

    # bleu = evaluate.load("bleu")
    # results = bleu.compute(predictions=dataset['predict'], references=dataset['label'])
    # print(results)
    # print("*"*20,"ALL DONE","*"*20)