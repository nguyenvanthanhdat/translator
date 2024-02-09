from datasets import load_dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from argparse import ArgumentParser



print("LOAD MODEL")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large")
model = PeftModel.from_pretrained(model, "lora/checkpoint-55000")
print("SAVE MODEL")
model.save_pretrained("mt5-translate")

print("LOAD MODEL FP16")
model = AutoModelForSeq2SeqLM.from_pretrained("mt5-translate", torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")


def trans(examples):

    # question
    question = ["en: " + i + "|END|" for i in examples['question']]
    question_tokened = tokenizer(question, return_tensors="pt", padding=True)
    question_len = len(question_tokened['input_ids'][0])
    question_outputs = model.generate(**question_tokened.to("cuda"), max_new_tokens=question_len+100, num_beams=3, use_cache=True, early_stopping=True)
    examples['question_vi'] =  tokenizer.batch_decode(question_outputs, skip_special_tokens=True)
    examples['question_vi'] = [i.split("|END|")[0][4:] for i in examples['question_vi']]

    # answer
    answer = ["en: " + i + "|END|" for i in examples["answer"]]
    answer_tokened = tokenizer(answer, return_tensors="pt", padding=True)
    answer_len = len(answer_tokened['input_ids'][0])
    answer_outputs = model.generate(**answer_tokened.to("cuda"), max_new_tokens=answer_len+200, num_beams=3, use_cache=True, early_stopping=True)
    examples['answer_vi'] = tokenizer.batch_decode(answer_outputs, skip_special_tokens=True)
    examples['answer_vi'] = [i.split("|END|")[0][4:] for i in examples['answer_vi']]

    # referecences
    temp = []
    for reference in examples['references']: # List[List]]
        new_reference = ["en: " + i + "|END|" for i in reference]
        new_reference_tokened = tokenizer(new_reference, return_tensors="pt", padding=True)
        new_reference_len = len(new_reference_tokened['input_ids'][0])
        new_reference_outputs = model.generate(**new_reference_tokened.to("cuda"), max_new_tokens=new_reference_len+100, num_beams=3, use_cache=True, early_stopping=True)
        temp_temp = tokenizer.batch_decode(new_reference_outputs, skip_special_tokens=True)
        temp_temp = [i.split("|END|")[0][4:] for i in temp_temp]
        temp.append(temp_temp)
    examples['references_vi'] = temp


    return examples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=100, required=False, type=int)
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(trans, batched=True, batch_size=args.batch_size)
    dataset.push_to_hub(f"presencesw/{args.dataset_name}_translated", token="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")