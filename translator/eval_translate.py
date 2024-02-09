from ctransformers import AutoModelForCausalLM
from argparse import ArgumentParser
from datasets import load_dataset



print("LOAD MODEL")
llm = AutoModelForCausalLM.from_pretrained(
    "presencesw/Vistral-7B-Chat",
    model_file="Vistral-7B-Chat-q5_0.gguf",
    model_type="mistral",
    gpu_layers=200,
    context_length = 6000,
#     hf=True
)

prompt = """
You are a translator.
Can you evaluate the mistakes in this translate from English to Vietnamese. Give the answer is "OK" or "not OK" and give a explanation after that:

"{en}"

"{vi}"
### Answer:
"""

def eval_llm(en, vi):
    _eval = ""
    while True:
        temp_prompt = prompt.format(en = en, vi = vi)
        _eval = llm(temp_prompt, temperature=0.5, max_new_tokens=100)
        if "### Explanation:" in _eval:
            _eval = _eval.split("### Explanation:")[0]
            if "not ok" in _eval.lower():
                return "False"
    return "True"

def eval_trans(example):
    question_en = example['question']
    question_vi = example['question_vi']
    answer_en = example['answer']
    answer_vi = example['answer_vi']
    references_en = example['references']
    references_vi = example['references_vi']

    if example["eval_question"] == "True":
        example['eval_question'] = eval_llm(question_en, question_vi)
    if example["eval_answer"] == "True":
        example['eval_answer'] = eval_llm(answer_en, answer_vi)
    list_eval_references = []
    for i in range(len(references_en)):
        if example['eval_references'][i] == "True":
            list_eval_references.append(eval_llm(references_en[i], references_vi[i]))
    example['eval_references'] = list_eval_references

    return example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=100, required=False, type=int)
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(eval_trans)
    dataset.save_to_disk("{args.dataset_name}")
    dataset.push_to_hub(f"{args.dataset_name}", token="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")