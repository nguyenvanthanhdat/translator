import os
import copy
import torch
import gdown
from argparse import ArgumentParser

import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
from accelerate import PartialState
from peft import PeftModel



def preprocess(examples, language_a, language_b):
    examples['input'] = [f'{language_a}: {sample}<END>' for sample in examples[language_a]]
    examples['label'] = [f'{language_b}: {sample}<END>' for sample in examples[language_b]]
    return examples

def tokenize(examples, token, max_length):
    prefix = [exp.strip() for exp in examples['input']]
    inputs = token(
        prefix, return_tensors="pt",
        padding="max_length", truncation=True, max_length=max_length
    )
    examples['input_ids'] = inputs['input_ids']
    examples['attention_mask'] = inputs['attention_mask']
    return examples

def get_output(examples, model, tokenizer, max_length, num_beams):
    # prefix = [exp.strip() for exp in examples['input']]
    # inputs = tokenizer(
    #     prefix, return_tensors="pt",
    #     padding="max_length", truncation=True, max_length=max_length
    # ).to("cuda")
    inputs = copy.deepcopy(examples)
    # print(type(inputs))
    # print(inputs)
    # inptus = dict(inputs)
    inputs.pop('len')
    inputs.pop('input')
    inputs.pop('label')
    inputs = {key: torch.tensor(inputs[key]).to('cuda') for key in inputs}
    outputs = model.generate(**inputs,
                             max_new_tokens=max_length,
                             num_beams=num_beams,
                             use_cache=True,
                             early_stopping=True)
    # outputs = [output[0] for output in outputs]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    examples['predict'] = outputs
    return examples

def postprocess(examples):
    for exp in examples['predict']:
        if exp[:4] in ["vi: ", "en: "]:
            exp = exp[4:]
        exp = exp.split("<END>")[0]
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
    parser.add_argument('--max_length', default=512, required=False, type=int)
    parser.add_argument('--num_beams', default=5, required=False)
    parser.add_argument('--gdown_id', default=None, required=False)
    parser.add_argument('--use_lora', default=None, required=True)
    parser.add_argument('--lora_path', default=None, required=True, type=bool)
    args = parser.parse_args()

    path = os.getcwd()
    # eval_path = os.path.join(os.getcwd(), "/eval")

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

    if args.gdown_id is not None:
        args.model_name_or_path = gdown.download(id = args.gdown_id)
        os.system(f"unzip -n {args.model_name_or_path} -d .")
        args.model_name_or_path = args.model_name_or_path.split(".")[0]


    # Load tokenizer
    if os.path.isdir(os.path.join(path, args.tokenizer_name_or_path)):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, args.model_name_or_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    distributed_state = PartialState()
    with distributed_state.split_between_processes(["en->vi", "vi->en"]) as distribute:
        distribute = distribute[0].split("->")
        language_a = distribute[0]
        language_b = distribute[1]

        if os.path.isdir(os.path.join(path, args.model_name_or_path)):
            if args.use_lora:
                model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, args.model_name_or_path), torch_dtype=torch.float16)
                model = PeftModel.from_pretrained(model, "lora/checkpoint-55000", torch_dtype=torch.float16).to("cuda")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, args.model_name_or_path), torch_dtype=torch.float16).to('cuda')
                model = PeftModel.from_pretrained(model, "lora/checkpoint-55000", torch_dtype=torch.float16).to("cuda")
        else:
            if args.use_lora:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
                model = PeftModel.from_pretrained(model, "lora/checkpoint-55000", torch_dtype=torch.float16).to("cuda")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16).to('cuda')
        print("*"*20,f"Preprocess data","*"*20)
        preprocess_dataset = dataset.map(
            preprocess, remove_columns=["en", "vi"], batched=True,
            fn_kwargs={"language_a": language_a,"language_b":language_b}
        )
        token_dataset = preprocess_dataset.map(
            tokenize, batched=True, batch_size=2,
            fn_kwargs={"token": tokenizer, "max_length": args.max_length}
        )


        for num_beam in args.num_beams.split(","): # [3, 4, 5]
            print("*"*20,f"Translate with num_bema = {num_beam}, {language_a} -> {language_b} ...","*"*20)
            dataset_translated = token_dataset.map(get_output,
                        fn_kwargs={"tokenizer": tokenizer, "model": model,
                                "max_length": args.max_length, "num_beams": int(num_beam)},
                        batched=True,
                        batch_size=args.batch_size)
            print("*"*20,"Postprocess data","*"*20)
            # dataset_translated = dataset_translated.map(postprocess, batched=True)
            dataset_translated = dataset_translated.remove_columns(['len', "input_ids", "attention_mask"])
            # dataset_translated.to_json(os.path.join(eval_path,f"{language_a}{language_b}-beam{num_beam}.txt"))
            dataset_translated.to_json(f"eval/{language_a}{language_b}-beam{num_beam}.txt", force_ascii=False)

    os.chdir("eval")
    bleu = evaluate.load("bleu")
    score = open("score.txt", "w")
    for file_txt in os.listdir("."):
        dataset = load_dataset("json", file_txt, split='train')
        try:
            results = bleu.compute(predictions=dataset['predict'], references=dataset['label'])
        except:
            results = 0
    score.write(f"{file_txt}: bleu - {results}\n")
    score.close()
    os.chdir("..")
    os.system("zip -r result.zip eval")