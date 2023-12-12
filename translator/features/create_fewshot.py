import json
import argparse
from datasets import load_dataset

def main(args):
    num_8k = args.num // 2
    num_aqua = args.num - num_8k

    # load prompt
    prompts = open("zalo_ai_math/utils/prompt_vi.txt", "r", encoding="utf-8").read()
    prompts = prompts.split("\n\n\n")

    # load aqua
    aqua = load_dataset('json', data_files= 'data/raw/AQUA.vi.json')['train']
    aqua_question = aqua[0]['question']
    aqua_answer = aqua[0]['answer']
    aqua_cot = aqua[0]['cot']
    aqua_choice = aqua[0]['choice']

    # load 8k
    gsm8k = load_dataset('json', data_files= 'data/raw/GSM8k.vi.json')['train']
    gsm8k_question = gsm8k[0]['text']
    gsm8k_answer = gsm8k[0]['answer']
    gsm8k_cot = gsm8k[0]['cot']
    gsm8k_choice = ""

    # store data
    all_tasks = []
    for prompt in prompts:
        temp = {}
        prompt = prompt.split("\n\n")
        tasks = prompt[0]
        inputs = prompt[1]
        targets = prompt[2]
        count = 0
        prefix = ""
        for i in range(num_aqua):
            choices = aqua_choice[str(i)].split("\n")
            answer = choices[ord(aqua_answer[str(i)][1]) - ord("A")]
            input_format_aqua = {
                'question': aqua_question[str(i)],
                'answer': answer,
                'cot': aqua_cot[str(i)],
                'choice': aqua_choice[str(i)],
            }
            new_inputs = inputs.format(**input_format_aqua).replace("(", "").replace(")", ".")
            new_targets = targets.format(**input_format_aqua)
            prefix = prefix + f"Ví dụ {count + 1}: " + new_inputs + "\n" + new_targets + "\n\n"
            count += 1
            if count == args.num:
                break
            input_format_gsm8k = {
                'question': gsm8k_question[str(i)],
                'answer': gsm8k_answer[str(i)],
                'cot': gsm8k_cot[str(i)],
                'choice': "Không có",
            }
            new_inputs = inputs.format(**input_format_gsm8k)
            new_targets = targets.format(**input_format_gsm8k)
            prefix = prefix + f"Ví dụ {count + 1}: " + new_inputs + "\n" + new_targets + "\n\n"
            count += 1
            if count == args.num:
                break
        temp['prefix'] = prefix
        temp['tasks'] = tasks
        all_tasks.append(temp)
    
    # write fewshot interleaved
    with open(f'zalo_ai_math/utils/few_shot_{args.num}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(all_tasks, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num',
        type=int,
        help="num few-shot. It show be even number")
    args = parser.parse_args()
    main(args=args)