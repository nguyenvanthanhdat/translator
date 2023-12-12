import torch
from peft import PeftModel, PeftConfig, LoraConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM)
from adapters import AutoAdapterModel
from datasets import load_dataset
from argparse import ArgumentParser
import os

# mode_or_path = 'bigscience/mt0-large'
# adapters_path = 'D:\Workspace\zalo_ai_math\output\\finetune\checkpoint-10'


EXPL_PROMPT = """Please act as an expert in multi-lingual understanding Vietnamese.
Request: $REQUEST.
Let's understand the task in English step by step!"""
PRED_PROMPT = """After understanding, you should act as an expert in English. 
Let’s resolve the task you understand above step-by-step!
Finally, you should format your answer as 'Answer: [num]'")"""

def get_output(examples, model, tokenizer):
    prefix = EXPL_PROMPT.replace("$REQUEST", examples['question'].strip() + " " + str(examples['choices']))
    # print(prefix + PRED_PROMPT)
    inputs = tokenizer.encode(prefix + PRED_PROMPT, return_tensors="pt").to("cuda")
    # inputs = tokenizer.encode(prefix + PRED_PROMPT, return_tensors="pt")
    outputs = model.generate(inputs, 
                             max_new_tokens=128,
                             num_beams=5,
                             early_stopping=True)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    examples['answer'] = outputs
    return examples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_test_path', default='.', required=True)
    parser.add_argument('--model_name_or_path', default='bigscience/mt0-large', required=True)
    parser.add_argument('--adapters_path', default=None, required=True)
    args = parser.parse_args()

    path = os.getcwd()
    
    dataset = load_dataset('json', data_files=os.path.join(path, args.data_test_path), split='train', field="data")
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, args.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, args.model_name_or_path))
    model = PeftModel.from_pretrained(model, args.adapters_path).to('cuda')
    # model = PeftModel.from_pretrained(model, args.adapters_path)
    dataset = dataset.map(get_output,
                fn_kwargs={"tokenizer": tokenizer, "model": model},
                remove_columns=['question', 'choices'])
    dataset.to_csv("result.csv")