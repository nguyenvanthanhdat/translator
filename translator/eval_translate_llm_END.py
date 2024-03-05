from ctransformers import AutoModelForCausalLM
from argparse import ArgumentParser
from datasets import load_dataset




print("LOAD MODEL")

# llm = AutoModelForCausalLM.from_pretrained(
#     "presencesw/Vistral-7B-Chat",
#     model_file="Vistral-7B-Chat-q5_0.gguf",
#     model_type="mistral",
#     gpu_layers=200,
#     context_length = 6000
# #     hf=True
# )

prompt = """
You are a translator.
Can you evaluate the mistakes in this translate from English to Vietnamese. Give the answer is "OK" or "not OK" and give a explanation after that:

"{en}"

"{vi}"
### Answer:
"""

def eval_llm(en, vi):
    if "END" in en or "END" in vi:
        return "False"
    return "True"

def eval_trans(example):
    question_en = example['question']
    question_vi = example['question_vi']
    answer_en = example['answer']
    answer_vi = example['answer_vi']
    references_en = example['references']
    references_vi = example['references_vi']

    example['eval_question'] = eval_llm(question_en, question_vi)
    example['eval_answer'] = eval_llm(answer_en, answer_vi)
    list_eval_references = []
    for i, j in zip(references_en, references_vi):
        list_eval_references.append(eval_llm(i, j))
    example['eval_references'] = list_eval_references

    return example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=100, required=False, type=int)
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(eval_trans)
    dataset.save_to_disk("{args.dataset_name}_translated_evaled")
    dataset.push_to_hub(f"{args.dataset_name}_translated_evaled", token="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")