from datasets import (
    load_dataset, 
    Dataset, 
    load_from_disk, 
    DatasetDict,
    concatenate_datasets
)
import argparse
import json

def main(args):
    # read prompt
    prompts = open("zalo_ai_math/utils/prompt_vi.txt", "r", encoding="utf-8").read()
    prompts = prompts.split("\n\n\n")

    # read prefix
    prefix = json.load(open(f"zalo_ai_math/utils/few_shot_{args.num}.json", "r", encoding="utf-8"))

    # read zalo data
    zalo = json.load(open("data/raw/ZALO.vi.json", "r", encoding="utf-8"))
    
    # create dataset
    data = []
    for sample in zalo['data']:
        question = sample['question']
        choice = sample['choices']
        try:
            cot = sample['explanation']
        except:
            cot = "Không có"
        answer = sample['answer']
        for i, prompt in enumerate(prompts):
            tasks, inputs, targets = prompt.split("\n\n") 
            input_format = {
                'question': question,
                'answer': answer,
                'cot': cot,
                'choice': choice,
            }
            new_inputs = inputs.format(**input_format)
            new_inputs = prefix[i]['prefix'] + new_inputs
            new_targets = targets.format(**input_format)
            input_dict = {
                "inputs": new_inputs,
                "targets": new_targets,
                "tasks": tasks
            }
            data.append(input_dict)
    zalo_dataset = Dataset.from_list(data)

    # read aqua.vi
    data = []
    aqua = json.load(open("data/raw/AQUA.vi.json", "r", encoding="utf-8"))
    question_aqua = aqua['question']
    choice_aqua = aqua['choice']
    answer_aqua = aqua['answer']
    cot_aqua = aqua['cot']
    for index in range(len(question_aqua)):
        for i, prompt in enumerate(prompts):
            tasks, inputs, targets = prompt.split("\n\n")
            input_format = {
                    'question': question_aqua[str(index)],
                    'answer': answer_aqua[str(index)],
                    'cot': cot_aqua[str(index)],
                    'choice': choice_aqua[str(index)],
                }
            new_inputs = inputs.format(**input_format)
            new_inputs = prefix[i]['prefix'] + new_inputs
            new_targets = targets.format(**input_format)
            input_dict = {
                "inputs": new_inputs,
                "targets": new_targets,
                "tasks": tasks
            }
            data.append(input_dict)

    aqua_dataset = Dataset.from_list(data)

    # read gms8k.vi
    data = []
    gsm8k = json.load(open("data/raw/GSM8K.vi.json", "r", encoding="utf-8"))
    question_gsm8k = gsm8k['text']
    answer_gsm8k = gsm8k['answer']
    cot_gsm8k = gsm8k['cot']
    for index in range(len(question_gsm8k)):
        for i, prompt in enumerate(prompts):
            tasks, inputs, targets = prompt.split("\n\n")
            input_format = {
                    'question': question_gsm8k[str(index)],
                    'answer': answer_gsm8k[str(index)],
                    'cot': cot_gsm8k[str(index)],
                    'choice': "Không có",
                }
            new_inputs = inputs.format(**input_format)
            new_inputs = prefix[i]['prefix'] + new_inputs
            new_targets = targets.format(**input_format)
            input_dict = {
                "inputs": new_inputs,
                "targets": new_targets,
                "tasks": tasks
            }
            data.append(input_dict)

    gsm8k_dataset = Dataset.from_list(data)
    dataset_strategy_1 = concatenate_datasets([zalo_dataset, aqua_dataset, gsm8k_dataset])
    dataset_strategy_1.save_to_disk("data/interim/strategy_1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num',
        type=int,
        help="num few-shot. It show be even number")
    args = parser.parse_args()
    main(args=args)