from datasets import load_dataset, Dataset
import random
import os

all_prompt = [
    ("{question} Let's think first. Chain of thought:",
     "{chain_of_thought}\nTherefore, the answer is {answer}."),
    ("{question} Think carefully first, then make a decision:",
     "{chain_of_thought} So, the answer is {answer}."),
    ("{question} Let's be accurate as possible.",
     "{chain_of_thought}\nThe answer: {answer}."),
    ("{question} Give me reasons, before answering the question",
     "{chain_of_thought} So the final answer is {answer}."),
    ("Lizzy: {question}.\nMe: Hmmm, let me think. I think this is the "
     "detailed solution:", "{chain_of_thought} Final answer: {answer}."),
    ("Question: {question} Think carefully first, then make a decision:",
     "{chain_of_thought} So the answer is {answer}."),
    ("Give the step-by-step reasoning process and then the final answer. "
     "{question}", "{chain_of_thought}\nThe final answer: {answer}."),
    ("{question}\nThoughts? Step-by-step reasoning:",
     "{chain_of_thought}\nThus, the answer is {answer}."),
    ("My question is: {question} Your thoughts:",
     "{chain_of_thought} The final answer: {answer}."),
    ("{question} Let's answer step by step:",
     "{chain_of_thought} The answer: {answer}."),
    ("Q: {question} Let's give some random thoughts before answering.",
     "{chain_of_thought}\nTherefore, the answer is {answer}."),
    ("{question} Hmmm, my stream of consciousness:",
     "{chain_of_thought} So, the answer is {answer}."),
    ("Give a quick stream of consciousness before answering the following "
     "question. {question}", "{chain_of_thought}\nThe answer: {answer}."),
    ("Use some thinking to answer the following question. {question}",
     "{chain_of_thought} So the final answer is {answer}."),
    ("Student: {question}.\nAnother student: Let's say, hmmm...\n",
     "{chain_of_thought} Final answer: {answer}."),
    ("{question} Think first, then make a decision. Some random thoughts:",
     "{chain_of_thought} So the answer is {answer}."),
    ("{question} Now, let's think a bit. Some random thoughts:",
     "{chain_of_thought}\nThe final answer: {answer}."),
    ("{question} Stream of consciousness:",
     "{chain_of_thought}\nThus, the answer is {answer}."),
    ("Question: {question} Random thoughts:",
     "{chain_of_thought} The final answer: {answer}."),
    ("{question} OK. Let's think. Some random thoughts first:",
     "{chain_of_thought} The answer: {answer}."),
    ("Give stream of consciousness and then the final answer. {question}",
     "{chain_of_thought}\nThe final answer: {answer}."),
    ("{question} Stream of consciousness first, then make a decision:",
     "{chain_of_thought}\nThus, the answer is {answer}."),
    ("Question: {question} Let's think first. Some random reasoning:",
     "{chain_of_thought} The final answer: {answer}."),
    ("Some question: {question}\nSome stream of consciousness:",
     "{chain_of_thought} The answer: {answer}."),
    ("{question} Let's think first. Stream of consciousness:",
     "{chain_of_thought}\nSo, the answer is {answer}."),
]

def add_prompt(examples):
    index = random.randint(0, 24)
    examples['prompt'] = all_prompt[index]
    return examples

def map_prompt(examples):
    features = {
        "question": examples['source'],
        "answer": examples['target'],
        "chain_of_thought": examples['rationale']
    }
    examples['inputs'] = examples['prompt'][0].format(**features)
    examples['targets'] = examples['prompt'][1].format(**features)
    return examples

def main():

    path = os.getcwd()

    # load dataset kai-cot
    dataset = load_dataset("json", data_files=os.path.join(path, "data/raw/pre-finetune/COT-KAI.json"))
    dataset = dataset['train']
    dataset = dataset.remove_columns(["task", "type"])

    random.seed(42)

    prompt_dataset = dataset.map(add_prompt)
    # prompt_dataset.to_json(os.path.join(path, "data/interim/pre-finetune/en_prompt.json"))

    en_dataset = prompt_dataset.map(map_prompt, remove_columns=prompt_dataset.column_names)
    # en_dataset_stage_1.to_json(os.path.join(path, "data/interim/pre-finetune/en_dataset.json"))
    for i in range(20):
        en_dataset = en_dataset.shuffle(seed=42)
    en_dataset = en_dataset.train_test_split(test_size=0.1)
    en_dataset['train'].to_json(os.path.join(path, "data/interim/pre-finetune/en_train.json"))
    en_dataset['test'].to_json(os.path.join(path, "data/interim/pre-finetune/en_test.json"))

if __name__ == "__main__":
    main()