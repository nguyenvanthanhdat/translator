from datasets import load_dataset, Dataset, concatenate_datasets
import random
import os
from tqdm import tqdm

all_prompt = {
    "cot+a->q": [
        ("Dựa vào các bước lập luận và câu trả lời,câu hỏi của các bước trên là gì? "
         "{chain_of_thought}\n Câu trả lời: {answer}", "The question {question}"
        ),
        ("Với chuỗi suy luận và câu trả lời này, câu hỏi có thể là gì?\n{chain_of_thought}\n A: {answer}", "Q: {question}"),
        ("Câu hỏi nào cho phần suy luận này và câu trả lời tương ứng "
         " answer?\n{chain_of_thought}\n Câu trả lời: {answer}",
         "Câu hỏi: {question}"),
    ],
    "q+a->cot": [
        ("Xem xét câu hỏi, {question}\n Quá trình lập luận từng bước để đến với câu trả lời: {answer}?",
         "{chain_of_thought}"),
        ("Câu hỏi. {question}\nCâu trả lời. {answer}\nQuá trình lập luận từng bước chứng minh cho câu trả lời đó là gì", "Các bước lập luận: {chain_of_thought}"),
        ("Q: {question}\nA: {answer}\nGiải thích cách đến được câu trả lời này ",
         "Giải thích: {chain_of_thought}"),
        ("Xem xét câu hỏi. {question}\n Nếu câu trả lời là '{answer}'; "
         "giải thích các bước suy luận:", "{chain_of_thought}"),
        ("Giải thích đơn giản tại sao {answer}là câu trả lời đúng cho câu hỏi: {question}. "
         "Giải thích:", "{chain_of_thought}"),
    ],
    "cot->q+a": [
        ("Dựa vào các lập luận, đưa ra một câu hỏi và một câu trả lời hợp lý. "
         "Suy luận theo từng bước {chain_of_thought}\n Câu hỏi"
         "và câu trả lời", "{question}\nCâu trả lời là: {answer}"),
        ("{chain_of_thought}\nĐiều này chứng minh cho câu hỏi và câu trả lời nào Q "
         "& A: ", "{question}\n{answer}"),
        ("{chain_of_thought}là các bước lập luận cho câu hỏi và câu trả lời nào?",
         "Q: {question}\nA: {answer}"),
        ("Dựa trên sự hiểu biết về các bước lập luận,, đưa ra một câu hỏi và câu trả lời hợp lý."
	     "Lý do: {chain_of_thought}\n Câu hỏi và câu trả lời" , "{question}\nCâu trả lời là: {answer}"),
        ("Lý do của các bước lập luận: {chain_of_thought}\nCặp câu hỏi và câu trả lời được mô tả dưới đây", "Q: {question}\nA: {answer}"),
        ("Tạo ra một cặp câu hỏi, câu trả lời từ những lời giải thích này: "
         "{chain_of_thought}\n", "Q:{question}\nA:{answer}"),
    ],
    "a->q+cot": [
        ("Đưa ra một câu hỏi và suy luận cho câu trả lời này "
         "Câu trả lời là: {answer}", "Câu hỏi là: {question}\n"
         "Quá trình lập luận theo từng bước: {chain_of_thought}\n"),
        ("Hãy tưởng tượng một câu hỏi và các bước lập luận cho câu trả lời này: "
         "{answer}", "Câu hỏi là {question}\nQuá trình lập luận theo từng bước "
         "{chain_of_thought}\n"),
        ("Đưa ra một câu hỏi và các bước lập luận chứng minh nó thuộc về câu hỏi này {answer}", "Câu hỏi là: {question}\n"
         "Các bước lập luận{chain_of_thought}\n"),
        ("Hãy tưởng tượng một câu hỏi và các bước lập luận cho câu trả lời này"
         " Đây là câu trả lời: {answer}", "Câu hỏi: {question}\n"
         "Giải thích các chuỗi lập luận: {chain_of_thought}"),
    ],
    "q->cot+a": [
        ("{question} Trước tiên, hãy suy nghĩ. Chuỗi suy luận :",
         "{chain_of_thought}\nVì thế, kết quả là {answer}."),
        ("{question} Trước tiên, suy nghĩ thật kỹ, sau đó hãy đưa ra quyết định:",
         "{chain_of_thought} Do đó, kết quả là {answer}."),
        ("{question} Hãy cố gắng chính xác nhất có thể.",
         "{chain_of_thought}\nĐáp án: {answer}."),
        ("{question} Hãy đưa ra lập luận trước khi đưa ra câu trả lời",
         "{chain_of_thought} Vì thế, đáp án cuối cùng là {answer}."),
        ("Lizzy: {question}.\nMe: Hmmm, để tôi suy nghĩ. Tôi nghĩ đây là giải pháp chi tiết:", "{chain_of_thought} Câu trả lời cuối cùng: {answer}."),
        ("Question: {question} Trước tiên, suy nghĩ thật kỹ, sau đó hãy đưa ra quyết định:",
         "{chain_of_thought} Vì thế, đáp án là: {answer}."),
        ("Hãy đưa ra quá trình lập luận từng bước rồi sau đó là câu trả lời cuối cùng. "
         "{question}", "{chain_of_thought}\nĐáp án cuối cùng: {answer}."),
        ("{question}\nHãy suy nghĩ? Lập luận theo từng bước:",
         "{chain_of_thought}\nDo đó, câu trả lời là {answer}."),
        ("Câu hỏi của tôi là: {question} Ý kiến của bạn thì sao:",
         "{chain_of_thought} Câu trả lời cuối cùng là: {answer}."),
        ("{question} Hãy trả lời theo từng bước:",
         "{chain_of_thought} Câu trả lời là: {answer}."),
    ],
    "q+cot->a": [
        ("{question} Trước tiên, hãy suy nghĩ. Chuỗi suy luận :"
         "{chain_of_thought}\n Vì thế, kết quả là ", "{answer}."),
        ("{question} Trước tiên, suy nghĩ thật kỹ, sau đó hãy đưa ra quyết định:"
         "{chain_of_thought} Do đó, kết quả là ", "{answer}."),
        ("{question} Hãy cố gắng chính xác nhất có thể."
         "{chain_of_thought}\nĐáp án: ", "{answer}."),
        ("{question} Hãy đưa ra lập luận trước khi đưa ra câu trả lời"
         "{chain_of_thought} Vì thế, đáp án cuối cùng là ", "{answer}."),
        ("Lizzy: {question}.\nMe: Hmmm, để tôi suy nghĩ. Tôi nghĩ đây là giải pháp chi tiết: {chain_of_thought} Câu trả lời cuối cùng: ", "{answer}."),
        ("Question: {question} Trước tiên, suy nghĩ thật kỹ, sau đó hãy đưa ra quyết định:"
         "{chain_of_thought} Vì thế, đáp án là: ", "{answer}."),
        ("Hãy đưa ra quá trình lập luận từng bước rồi sau đó là câu trả lời cuối cùng. "
         "{question} {chain_of_thought}\nĐáp án cuối cùng: ", "{answer}."),
        ("{question}\nHãy suy nghĩ? Lập luận theo từng bước:"
         "{chain_of_thought}\nDo đó, câu trả lời là ", "{answer}."),
        ("Câu hỏi của tôi là: {question} Ý kiến của bạn thì sao:"
         "{chain_of_thought} Câu trả lời cuối cùng là: ", "{answer}."),
        ("{question} Hãy trả lời theo từng bước:"
         "{chain_of_thought} Câu trả lời là: ", "{answer}."),
    ]
}

def map_prompt(examples):
    feature = None
    if "choice" in examples and examples['choice'] != "":
        if examples['type'] in ["q+a->cot", "q->cot+a", "q+cot->a"]:
            question = examples['question'] + " " + examples['choice']      
            features = {
                "question": question,
                "answer": examples['answer'],
                "chain_of_thought": examples['cot']
            }
        else:
            features = {
                "question": examples['question'],
                "answer": examples['answer'],
                "chain_of_thought": examples['cot']
            }
    elif "passage" in examples:
        question = examples['passage'] + " " +examples['question']
        features = {
            "question": question,
            "answer": examples['answer'],
            "chain_of_thought": examples['cot']
        }
    else:
        features = {
            "question": examples['question'],
            "answer": examples['answer'],
            "chain_of_thought": examples['cot']
        }
    examples['inputs'] = examples['prompt'][0].format(**features)
    examples['targets'] = examples['prompt'][1].format(**features)
    return examples

def main():

    path = os.getcwd()

    random.seed(42)
    all_dataset = []
    for file_name in os.listdir(os.path.join(path, "data/raw/pre-finetune/")):
        if file_name[-7:] == "vi.json" or file_name == "zalo_without_expl.json":
            dataset = load_dataset("json", data_files=os.path.join(path, "data/raw/pre-finetune/", file_name))
            dataset = dataset['train']
            try:
                dataset = dataset.rename_column("explanation", "cot")
            except:
                pass
            try:
                dataset = dataset.rename_column("option", "choice")
            except:
                pass
            if "cot" not in dataset.column_names:
                new_column = ["Không cần thiết"] * len(dataset)
                dataset = dataset.add_column("cot", new_column)
            temp = []
            for i in tqdm(range(len(dataset))):
                sample = dataset[i]
                for task in all_prompt:
                    index = random.randint(0, len(all_prompt[task]) - 1)
                    sample['prompt'] = all_prompt[task][index]
                    sample['type'] = task
                    temp.append(sample)
            dataset = Dataset.from_list(temp)
            
            # return
            # dataset = dataset.map(add_prompt)
            dataset = dataset.map(map_prompt, remove_columns=dataset.column_names)
            all_dataset.append(dataset)
            print(file_name)
            print(dataset)
    vi_dataset = concatenate_datasets(all_dataset)
    for i in range(20):
        vi_dataset = vi_dataset.shuffle(seed=42)
            # break
    vi_dataset = vi_dataset.train_test_split(test_size=0.1)
    vi_dataset['train'].to_json(os.path.join(path, "data/interim/pre-finetune/vi_train.json"))
    vi_dataset['test'].to_json(os.path.join(path, "data/interim/pre-finetune/vi_test.json"))
    return


if __name__ == "__main__":
    main()